import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
a = 0.3445
R_max = 5.0
N_t = 25
t_nodes = torch.linspace(0, 1, N_t).to(device)
dt = t_nodes[1] - t_nodes[0]

# Learnable radius values, initialized from R(0)=0.05
R_values = nn.Parameter(torch.linspace(0.05, 0.05, N_t).to(device))



# PINN for phi(r, t)
class PhiNet(nn.Module):
    def __init__(self):
        super(PhiNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 1)
        )

    def forward(self, r, t):
        return self.net(torch.cat([r, t], dim=1))


phi_net = PhiNet().to(device)


# Interpolation of R(t)
def interpolate_R(t_batch):
    t_batch = t_batch.squeeze()
    indices = torch.bucketize(t_batch, t_nodes, right=True)
    indices = torch.clamp(indices, 1, N_t - 1)
    t0 = t_nodes[indices - 1]
    t1 = t_nodes[indices]
    R0 = R_values[indices - 1]
    R1 = R_values[indices]
    weight = (t_batch - t0) / (t1 - t0)
    R_interp = R0 + weight * (R1 - R0)
    return R_interp.unsqueeze(1)


# Data generators
def get_collocation_points(N):
    t = torch.rand((N, 1), requires_grad=True).to(device)
    R = interpolate_R(t).detach()
    r = R + (R_max - R) * torch.rand((N, 1), requires_grad=True).to(device)
    return r, t


def get_initial_points(N):
    # r = torch.rand((N, 1), requires_grad=True).to(device) * (R_max)
    r = 0.0 + (R_max - 0.0) * torch.rand((N, 1), requires_grad=True).to(device)
    t = torch.zeros((N, 1), requires_grad=True).to(device)
    return r, t


def get_boundary_points(N):
    t = torch.rand((N, 1), requires_grad=True).to(device)
    R = interpolate_R(t).detach()
    r = R.clone().detach().requires_grad_()
    return r, t


# PDE residual
def pde_residual(r, t):
    phi = phi_net(r, t)
    phi_r = autograd.grad(phi, r, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_rr = autograd.grad(phi_r, r, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_t = autograd.grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    return (phi_t - phi_rr - 2 / (r) * phi_r) * torch.tanh(r) ** 2


# Radius dynamics loss
def radius_dynamics_loss():
    loss_R = 0.0
    for i in range(N_t - 1):
        t_i = t_nodes[i].reshape(1, 1).to(device).requires_grad_(True)
        R_i = R_values[i]
        R_ip1 = R_values[i + 1]
        dRdt = (R_ip1 - R_i) / dt
        r_probe = R_i.detach()
        r_probe = torch.tensor([[r_probe]], dtype=torch.float32, requires_grad=True).to(device)
        phi = phi_net(r_probe, t_i)
        dphi_dr = autograd.grad(phi, r_probe, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        loss_R += (dRdt - a * dphi_dr) ** 2
    return loss_R / (N_t - 1)


# Optimizer
optimizer = torch.optim.Adam(list(phi_net.parameters()) + [R_values], lr=1e-4)
loss_history = []

loss_total_hist = []
loss_pde_hist = []
loss_ic_hist = []
loss_bc_hist = []
loss_R_hist = []
loss_r0_hist = []

# Training loop
for epoch in range(75000):
    optimizer.zero_grad()
    r_c, t_c = get_collocation_points(1024)
    loss_pde = torch.mean(pde_residual(r_c, t_c) ** 2)
    r0, t0 = get_initial_points(1024)
    phi0 = phi_net(r0, t0)
    loss_ic = torch.mean((phi0 - 1) ** 2)
    rb, tb = get_boundary_points(1024)
    phib = phi_net(rb, tb)
    loss_bc = torch.mean(phib ** 2)
    loss_R = radius_dynamics_loss()
    loss_r0 = (R_values[0] - 0.0) ** 2
    loss = loss_pde + loss_ic + loss_bc + loss_R + loss_r0
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    loss_total_hist.append(loss.item())
    loss_pde_hist.append(loss_pde.item())
    loss_ic_hist.append(loss_ic.item())
    loss_bc_hist.append(loss_bc.item())
    loss_R_hist.append(loss_R.item())
    loss_r0_hist.append(loss_r0.item())

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.4e},  loss_ic: {loss_ic.item():.4e}, loss_R: {loss_R.item():.2e}, R0 Loss: {loss_r0.item():.2e}")
    if loss.item() < 1.0e-3:  # 5.0e-4: #5.0e-4:  #2.0e-3:
        print(f"Stopping early at epoch {epoch} — loss < 0.0002")
        break

loss_array = np.vstack([
    loss_total_hist,
    loss_pde_hist,
    loss_ic_hist,
    loss_bc_hist,
    loss_R_hist,
    loss_r0_hist
])
np.save("loss_history_components.npy", loss_array)
print("Saved loss history to loss_history_components.npy")

# Post-training: compare learned radius with Vasil'ev's expression
with torch.no_grad():
    R_learned = R_values.detach().cpu().numpy()
    t_vals = t_nodes.cpu().numpy()


    # Solve for b using integral equation: a = b * ∫₁^∞ dx/x² * exp[-(b/2)(x² - 1)]
    def integrand(x, b):
        return np.exp(-0.5 * b * (x ** 2 - 1)) / x ** 2


    def f_b(b):
        integral, _ = quad(integrand, 1, np.inf, args=(b,))
        return b * integral - a


    sol = root_scalar(f_b, bracket=[1e-6, 100], method='brentq')
    b_value = sol.root

    # Compute exact radius: R(t) = sqrt(2 * b * t)
    R_exact = np.sqrt(2 * b_value * t_vals)

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, R_learned, 'o-', label='PINN')
    plt.plot(t_vals, R_exact, '--', label=r'Exact: $\sqrt{2bt}$')
    plt.xlabel(r'$\tilde{t}$')
    plt.ylabel(r'$\tilde{R}(\tilde{t})$')
    plt.title(f"PINN vs Exact Radius from Vasil'ev\nb = {b_value:.4e}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig_R_vs_t.pdf', bbox_inches='tight')
    plt.show()


    def phi_vasiliev(rho_vals, b_val):
        I_norm, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, 1, np.inf)
        phi_vals = []
        for rho in rho_vals:
            if rho < 1:
                phi_vals.append(0.0)
            else:
                I_num, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, rho, np.inf)
                phi_vals.append(I_num / I_norm)
        return np.array(phi_vals)


    # Post-training: compare learned radius with Vasil'ev's expression
    with torch.no_grad():
        R_learned = R_values.detach().cpu().numpy()
        t_vals = t_nodes.cpu().numpy()


        def integrand(x, b):
            return np.exp(-0.5 * b * (x ** 2 - 1)) / x ** 2


        def f_b(b):
            integral, _ = quad(integrand, 1, np.inf, args=(b,))
            return b * integral - a


        sol = root_scalar(f_b, bracket=[1e-6, 100], method='brentq')
        b_value = sol.root
        print(f"b = {b_value:.5f}")

        np.save(f"R_vs_t_R0_{R_learned[0]:.3f}.npy", np.vstack([t_vals, R_learned]))
        print(f"Saved radius to: R_vs_t_R0_{R_learned[0]:.3f}.npy")


        def phi_vasiliev(rho_vals, b_val):
            I_norm, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, 1, np.inf)
            phi_vals = []
            for rho in rho_vals:
                if rho < 1:
                    phi_vals.append(0.0)
                else:
                    I_num, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, rho, np.inf)
                    phi_vals.append(1.0 - I_num / I_norm)
            return np.array(phi_vals)


        times = [0.10, 0.25, 0.50]

        R_learned_np = R_learned
        t_vals_np = t_vals

        for t_fixed in times:
            # интерполируем радиус в этот момент
            R_fixed = np.interp(t_fixed, t_vals_np, R_learned_np)

            # сетка r от R(t) до R_max
            r_test = np.linspace(R_fixed, R_max, 400)
            rho_vals = r_test / R_fixed
            t_test = np.full_like(r_test, t_fixed)

            # PINN-профиль
            r_tensor = torch.tensor(r_test.reshape(-1, 1), dtype=torch.float32).to(device)
            t_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32).to(device)
            phi_pred = phi_net(r_tensor, t_tensor).cpu().numpy().flatten()

            # Васильевский профиль
            phi_exact = phi_vasiliev(rho_vals, b_value)

            # Сохраняем в npy: r, phi_PINN, phi_Vasiliev
            profile_data = np.vstack([r_test, phi_pred, phi_exact])
            np.save(f"profile_R0_{R_learned_np[0]:.3f}_t{t_fixed:.2f}.npy", profile_data)
            print(f"Saved profile to: profile_R0_{R_learned_np[0]:.3f}_t{t_fixed:.2f}.npy")

        Nt = 50
        Nr = 200
        t_grid = np.linspace(0.05, 1.0, Nt)
        r_grid = np.linspace(0.0, 1.5, Nr)

        T, R = np.meshgrid(t_grid, r_grid, indexing='ij')


        def phi_vasiliev(rho_vals, b_val):
            I_norm, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, 1, np.inf)
            phi_vals = []
            for rho in rho_vals:
                if rho < 1:
                    phi_vals.append(0.0)
                else:
                    I_num, _ = quad(lambda x: np.exp(-0.5 * b_val * (x ** 2 - 1)) / x ** 2, rho, np.inf)
                    phi_vals.append(I_num / I_norm)
            return np.array(phi_vals)


        phi_pinn = np.zeros_like(T)

        with torch.no_grad():
            for i in range(Nt):
                t_val = t_grid[i]
                # интерполируем R(t)
                R_learned_np = R_values.detach().cpu().numpy()
                t_nodes_np = t_nodes.detach().cpu().numpy()
                R_t = np.interp(t_val, t_nodes_np, R_learned_np)

                r_line = r_grid.copy()
                # далее считаем только вне капли
                mask = r_line >= R_t
                r_valid = r_line[mask]

                t_batch = torch.full((r_valid.size, 1), t_val, dtype=torch.float32).to(device)
                r_batch = torch.tensor(r_valid.reshape(-1, 1), dtype=torch.float32).to(device)

                phi_batch = phi_net(r_batch, t_batch).cpu().numpy().flatten()

                phi_pinn[i, mask] = phi_batch
                phi_pinn[i, ~mask] = 0.0

        phi_vas = np.zeros_like(T)

        for i in range(Nt):
            t_val = t_grid[i]
            # радиус по Васильеву
            R_t = np.sqrt(2 * b_value * t_val)
            r_line = r_grid.copy()
            mask = r_line >= R_t
            rho = r_line[mask] / R_t
            phi_vas_line = 1.0 - phi_vasiliev(rho, b_value)  # как ты делал в профилях

            phi_vas[i, mask] = phi_vas_line
            phi_vas[i, ~mask] = 0.0

        err = np.abs(phi_pinn - phi_vas)

        finite_vals = err[np.isfinite(err)]
        vmax = np.percentile(finite_vals, 99) if finite_vals.size > 0 else None

        # ---- кастомный colormap ----
        from matplotlib.colors import LinearSegmentedColormap

        colors = [
            (1.0, 1.0, 1.0),  # white
            (1.0, 0.8, 0.8),  # light red
            (0.9, 0.3, 0.3),  # medium red
            (0.6, 0.0, 0.0)  # dark red
        ]
        cmap_err = LinearSegmentedColormap.from_list("white_to_red", colors)

        fig, ax = plt.subplots(figsize=(12, 5))

        im = ax.pcolormesh(
            t_grid,
            r_grid,
            err.T,
            shading='auto',
            cmap=cmap_err,
            vmin=0.0,
            vmax=vmax
        )

        axis_fontsize = 24
        tick_fontsize = 20
        cbar_label_fontsize = 24

        ax.set_xlabel(r'$\tilde{t}$', fontsize=axis_fontsize)
        ax.set_ylabel(r'$\tilde{r}$', fontsize=axis_fontsize)

        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # ---- цветовая шкала ----
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(r'$|\varphi_{\mathrm{PINN}} - \varphi_{\mathrm{SS}}|$', fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=tick_fontsize)

        plt.tight_layout()
        plt.savefig("fig_error_heatmap.pdf", bbox_inches='tight', dpi=300)
        plt.show()