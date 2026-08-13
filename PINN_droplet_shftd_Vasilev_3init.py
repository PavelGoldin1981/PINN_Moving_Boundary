#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== PINN for diffusion to a growing sphere: three initial radii =====
# Trains a PINN for R(t) three times (different R0), plots:
# 1) R(t) for all three runs + asymptote sqrt(2 b t)
# 2) R^2(t) - 2 b t (should -> R0^2 at late times)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

# ---------------- Plot style (as in your example) ----------------
style = {
    'figure.figsize': (10, 8),
    'font.size': 36,
    'axes.labelsize': '26',
    'axes.titlesize': '26',
    'xtick.labelsize': '26',
    'ytick.labelsize': '26',
    'legend.fontsize': '20',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 3,
    'font.family': 'serif',
    'text.usetex': False,
    'figure.autolayout': True
}
matplotlib.rcParams.update(style)

# ---------------- Output dir ----------------
os.makedirs('figs', exist_ok=True)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Global hyperparams (shared) ----------------
a = 0.1          # dimensionless supersaturation in Stefan condition
R_max = 7.0      # far boundary radius (dimensionless)
N_t = 25         # time nodes
t_nodes = torch.linspace(0, 1, N_t).to(device)
dt = float(t_nodes[1] - t_nodes[0])

# For reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# ---------------- Network ----------------
class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 15), nn.Tanh(),
            nn.Linear(15, 15), nn.Tanh(),
            nn.Linear(15, 15), nn.Tanh(),
            nn.Linear(15, 1)
        )
    def forward(self, r, t):
        return self.net(torch.cat([r, t], dim=1))

# ---------------- Helpers ----------------
def interpolate_R(t_batch, t_nodes, R_values):
    """Linear interpolation of R(t) on training nodes."""
    t_batch = t_batch.squeeze()
    idx = torch.bucketize(t_batch, t_nodes, right=True)
    idx = torch.clamp(idx, 1, len(t_nodes)-1)
    t0, t1 = t_nodes[idx-1], t_nodes[idx]
    R0, R1 = R_values[idx-1], R_values[idx]
    w = (t_batch - t0) / (t1 - t0)
    return (R0 + w * (R1 - R0)).unsqueeze(1)

def get_collocation_points(N, t_nodes, R_values):
    t = torch.rand((N,1), requires_grad=True).to(device)
    R = interpolate_R(t, t_nodes, R_values).detach()
    r = R + (R_max - R) * torch.rand((N,1), requires_grad=True).to(device)
    return r, t

def get_initial_points(N, R_values):
    """IC points sampled in [R(0), R_max] to stay outside the droplet for any R0."""
    R_start = R_values[0].detach()
    r = R_start + (R_max - R_start) * torch.rand((N,1), requires_grad=True).to(device)
    t = torch.zeros((N,1), requires_grad=True).to(device)
    return r, t

def get_boundary_points(N, t_nodes, R_values):
    """Points on the moving boundary r=R(t)."""
    t = torch.rand((N,1), requires_grad=True).to(device)
    R = interpolate_R(t, t_nodes, R_values).detach()
    r = R.clone().detach().requires_grad_()
    return r, t

def get_far_points(N):
    """Optional far boundary r=R_max -> 0 (dimensionless)."""
    t = torch.rand((N,1), requires_grad=True).to(device)
    r = torch.full((N,1), R_max, requires_grad=True).to(device)
    return r, t

def pde_residual(phi_net, r, t):
    phi = phi_net(r, t)
    phi_r  = autograd.grad(phi, r, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_rr = autograd.grad(phi_r, r, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_t  = autograd.grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    # smooth factor near small r (not critical here since r>=R(t) >= R0 > 0)
    return (phi_t - phi_rr - 2.0/r * phi_r) * torch.tanh(r)**2

def radius_dynamics_loss(phi_net, R_values, t_nodes, a):
    loss_R = 0.0
    for i in range(len(t_nodes)-1):
        t_i = t_nodes[i].reshape(1,1).to(device).requires_grad_(True)
        R_i, R_ip1 = R_values[i], R_values[i+1]
        dRdt = (R_ip1 - R_i) / dt
        # evaluate gradient at the interface r=R_i
        r_probe = torch.tensor([[float(R_i.detach().cpu())]], dtype=torch.float32, requires_grad=True).to(device)
        phi = phi_net(r_probe, t_i)
        dphi_dr = autograd.grad(phi, r_probe, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        loss_R = loss_R + (dRdt - a * dphi_dr)**2
    return loss_R / (len(t_nodes)-1)

# ---------------- One training run for given R0 ----------------
def train_one_R0(R0, epochs=20000, lr=1e-4, N_c=1024, add_far_bc=False,
                 early_stop=2e-3, seed=1234, verbose=True):
    # separate seeds per run
    torch.manual_seed(seed)
    np.random.seed(seed)

    phi_net = PhiNet().to(device)
    # initialize R(t): small ramp from R0 to R0+0.1
    R_values = nn.Parameter(torch.linspace(R0, R0+0.1, N_t, device=device))
    optimizer = torch.optim.Adam(list(phi_net.parameters()) + [R_values], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE collocation
        r_c, t_c = get_collocation_points(N_c, t_nodes, R_values)
        loss_pde = torch.mean(pde_residual(phi_net, r_c, t_c)**2)

        # Initial condition: phi(r,0) = 1 on [R(0), R_max]
        r0, t0 = get_initial_points(N_c, R_values)
        phi0 = phi_net(r0, t0)
        loss_ic = torch.mean((phi0 - 1.0)**2)

        # Moving boundary: phi(R(t),t) = 0
        rb, tb = get_boundary_points(N_c, t_nodes, R_values)
        phib = phi_net(rb, tb)
        loss_bc = torch.mean(phib**2)

        # Optional far boundary: phi(R_max, t) = 0 (stabilizes)
        loss_far = 0.0
        if add_far_bc:
            rf, tf = get_far_points(N_c//2)
            phif = phi_net(rf, tf)
            loss_far = torch.mean(phif**2)

        # Stefan coupling: dR/dt = a * phi_r |_{r=R(t)}
        loss_R = radius_dynamics_loss(phi_net, R_values, t_nodes, a)

        # enforce initial radius
        loss_r0 = (R_values[0] - R0)**2

        loss = loss_pde + loss_ic + loss_bc + loss_R + loss_r0 + loss_far
        loss.backward()
        optimizer.step()

        if verbose and epoch % 500 == 0:
            print(f"[R0={R0:.3f}] epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(PDE {loss_pde.item():.2e}, BC {loss_bc.item():.2e}, R {loss_R.item():.2e}, FAR {float(loss_far):.2e})")

        if loss.item() < early_stop:
            if verbose:
                print(f"[R0={R0:.3f}] early stop at epoch {epoch}, loss={loss.item():.3e}")
            break

    # learned R(t)
    R_np = R_values.detach().cpu().numpy()
    t_np = t_nodes.detach().cpu().numpy()
    return t_np, R_np

# ---------------- Solve for b from the implicit integral equation ----------------
def solve_b_from_a(a):
    """Solve a = b * ∫_1^∞ e^{- (b/2) (x^2 - 1)} / x^2 dx"""
    def integrand(x, b):
        return np.exp(-0.5 * b * (x**2 - 1)) / (x**2)
    def F(b):
        val, _ = quad(integrand, 1.0, np.inf, args=(b,))
        return b * val - a
    sol = root_scalar(F, bracket=[1e-8, 100.0], method='brentq')
    return sol.root

# ---------------- Run three trainings ----------------
R0_list = [0.02, 0.05, 0.10]
results = []
for j, R0 in enumerate(R0_list):
    t_np, R_np = train_one_R0(R0, epochs=20000, lr=1e-4, N_c=1024,
                              add_far_bc=False, early_stop=2e-3,
                              seed=1234+j, verbose=True)
    results.append((R0, t_np, R_np))

# ---------------- Compute asymptotic law sqrt(2 b t) ----------------
b_value = solve_b_from_a(a)
print(f"b (from implicit integral) = {b_value:.6e}")

# choose a common fine time grid for asymptote (0..1)
t_asym = results[0][1]  # reuse t_nodes
R_asym = np.sqrt(2.0 * b_value * t_asym)

# ---------------- Plot 1: R(t) three runs + asymptote ----------------
fig, ax = plt.subplots(1,1)
for (R0, t_np, R_np) in results:
    ax.plot(t_np, R_np, label=fr'PINN $R_0={R0}$')
ax.plot(t_asym, R_asym, 'k--', label=r'Asymptote $\sqrt{2 b t}$')
ax.set_xlabel(r'$\tilde{t}$')
ax.set_ylabel(r'$\tilde{R}(\tilde{t})$')
ax.set_title('PINN-learned radius for three initial radii')
ax.legend()
fig.savefig('figs/fig_R_three_PINN.pdf', bbox_inches='tight')
plt.show()

# ---------------- Plot 2: R^2 - 2 b t (should -> R0^2) ----------------
fig2, ax2 = plt.subplots(1,1)
for (R0, t_np, R_np) in results:
    ax2.plot(t_np, R_np**2 - 2.0*b_value*t_np, label=fr'$R_0={R0}$')
ax2.set_xlabel(r'$\tilde{t}$')
ax2.set_ylabel(r'$\tilde{R}^{\,2} - 2 b \tilde{t}$')
ax2.set_title(r'Late-time limit $\to R_0^2$')
ax2.legend()
fig2.savefig('figs/fig_R2_minus_2bt_three.pdf', bbox_inches='tight')
plt.show()