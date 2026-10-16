# Mathematical Formulation: Multi-Body Astrodynamics & Symplectic PINNs

## 1. Coupled Perturbed Equations of Motion in GCRF
In the Geocentric Celestial Reference Frame (GCRF), the satellite state $\mathbf{x} = [\mathbf{r}^\top, \mathbf{v}^\top]^\top \in \mathbb{R}^6$ evolves under:

$$\ddot{\mathbf{r}} = -\frac{\mu_E}{r^3}\mathbf{r} + \mathbf{a}_{J_2\text{--}J_4}(\mathbf{r}) + \mathbf{a}_{\text{drag}}(\mathbf{r}, \mathbf{v}, F_{10.7}, K_p) + \mathbf{a}_{\text{SRP}}(\mathbf{r}, \mathbf{r}_\odot) + \mathbf{a}_{3B}(\mathbf{r}, \mathbf{r}_M, \mathbf{r}_\odot)$$

### 1.1 Zonal Harmonics ($J_2, J_3, J_4$)
With $\zeta = z / r$ and equatorial radius $R_E$:
$$\mathbf{a}_{J_2} = -\frac{3}{2} J_2 \frac{\mu_E R_E^2}{r^5} \begin{bmatrix} x(1 - 5\zeta^2) \\ y(1 - 5\zeta^2) \\ z(3 - 5\zeta^2) \end{bmatrix}$$

### 1.2 Circular Restricted Three-Body Problem (CR3BP) & Jacobi Constant
In the barycentric rotating synodic frame with mass parameter $\mu = \frac{\mu_M}{\mu_E + \mu_M}$:
$$\Omega(x, y, z) = \frac{1}{2}(x^2 + y^2) + \frac{1 - \mu}{r_1} + \frac{\mu}{r_2}$$
$$C_J = 2\Omega(x, y, z) - (\dot{x}^2 + \dot{y}^2 + \dot{z}^2)$$

## 2. Canonical Non-Dimensionalization
To eliminate gradient pathology across 7 orders of magnitude between position ($10^7\text{ m}$) and velocity ($10^3\text{ m/s}$):
$$R_{\text{scale}} = R_E, \quad V_{\text{scale}} = \sqrt{\frac{\mu_E}{R_E}}, \quad \tau_{\text{scale}} = \sqrt{\frac{R_E^3}{\mu_E}}$$
$$\mathbf{r}^* = \frac{\mathbf{r}}{R_{\text{scale}}}, \quad \mathbf{v}^* = \frac{\mathbf{v}}{V_{\text{scale}}}, \quad \tau = \frac{t}{\tau_{\text{scale}}}$$

## 3. Physics-Informed Residual Loss & Symplectic Hamiltonian Projection
The PINN predicts the normalized state transition $\hat{\mathbf{s}}_{k+1} = \mathbf{s}_k + \mathcal{N}_\theta(\mathbf{s}_k, \mathbf{p}_k)$ conditioned on $\mathbf{p}_k = [B, F_{10.7}, K_p, \Delta \tau]^\top$.
On Core 0 of ESP32 Node 2, the predicted velocity $\mathbf{v}_{\text{NN}}$ is projected onto the target Hamiltonian manifold $E_{\text{target}} = E_k + \Delta E_{\text{drag}}$:

$$v_{\text{target}} = \sqrt{2\left(E_{\text{target}} + \frac{\mu_E}{\|\mathbf{r}_{k+1}\|}\right)}, \quad \mathbf{v}_{k+1} = \mathbf{v}_{\text{NN}} \frac{v_{\text{target}}}{\|\mathbf{v}_{\text{NN}}\|}$$
