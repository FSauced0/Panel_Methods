import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# LOAD GEOMETRY
# =============================================================================
df = pd.read_csv(r'C:\Users\fabia\Downloads\v23010.dat', sep=r'\s+', skiprows=1)

XB = df.iloc[:, 0].values.astype(float)
YB = df.iloc[:, 1].values.astype(float)

# Remove duplicate closing point if present
if np.isclose(XB[-1], XB[0]) and np.isclose(YB[-1], YB[0]):
    XB = XB[:-1]
    YB = YB[:-1]

numPan = len(XB) - 1
print(f"Number of panels: {numPan}")

# =============================================================================
# ENFORCE COUNTER-CLOCKWISE (CCW) ORDERING
# Shoelace sum negative = CCW
# =============================================================================
edge = np.zeros(numPan)
for i in range(numPan):
    edge[i] = (XB[i+1] - XB[i]) * (YB[i+1] + YB[i])
sumEdge = np.sum(edge)

print(f"\nsumEdge = {sumEdge:.4f}")
if sumEdge > 0:
    XB = np.flipud(XB)
    YB = np.flipud(YB)
    print("Flipped to CW ordering.")
else:
    print("Already CW ordering.")

# =============================================================================
# ANGLE OF ATTACK
# =============================================================================
AoA   = 0
AoA_r = np.deg2rad(AoA)
Vinf  = 1.0
Ux    = Vinf * np.cos(AoA_r)
Uy    = Vinf * np.sin(AoA_r)

# =============================================================================
# PANEL GEOMETRY
# =============================================================================
XC  = np.zeros(numPan)
YC  = np.zeros(numPan)
S   = np.zeros(numPan)
phi = np.zeros(numPan)

for i in range(numPan):
    XC[i]  = 0.5 * (XB[i] + XB[i+1])          # control point x
    YC[i]  = 0.5 * (YB[i] + YB[i+1])          # control point y
    dx     = XB[i+1] - XB[i]
    dy     = YB[i+1] - YB[i]
    S[i]   = np.sqrt(dx**2 + dy**2)            # panel length
    phi[i] = np.arctan2(dy, dx)                # panel tangent angle [rad]

# Wrap phi to [0, 2*pi]
phi = np.where(phi < 0, phi + 2*np.pi, phi)

# For CCW ordering: outward normal is +90 deg from tangent
# tangent = (cos phi, sin phi)
# outward normal = (sin phi, -cos phi)
nx = np.sin(phi)
ny = -np.cos(phi)

# =============================================================================
# KATZ & PLOTKIN SOURCE PANEL INFLUENCE INTEGRALS
# I(i,j) = normal influence   (goes into A matrix)
# J(i,j) = tangential influence (goes into Vt computation)
#
# In panel j local frame (xstar, ystar):
#   I = [ ystar*(th1-th2) + (xstar)*ln(r1/r2) - S[j]*ln(r2) + S[j]*ln(r1) ]
# Simplified closed-form per Katz & Plotkin eq 11.26:
#   Cn (normal)     = sin(phi_i - phi_j)*ln(r1/r2) + cos(phi_i - phi_j)*(th1-th2)
#   Ct (tangential) = cos(phi_i - phi_j)*ln(r1/r2) - sin(phi_i - phi_j)*(th1-th2)
# =============================================================================
# Pre-allocate
CN = np.zeros((numPan, numPan))   # normal influence
CT = np.zeros((numPan, numPan))   # tangential influence

for i in range(numPan):
    for j in range(numPan):
        if i == j:
            CN[i, j] = 0.0        # self normal = 0 (added as 1 on diagonal via BCs)
            CT[i, j] = 0.0        # self tangential = 0
            continue

        # Translate CP i into panel j local frame
        xt =  XC[i] - XB[j]
        yt =  YC[i] - YB[j]
        xstar =  xt*np.cos(phi[j]) + yt*np.sin(phi[j])
        ystar = -xt*np.sin(phi[j]) + yt*np.cos(phi[j])

        r1 = np.hypot(xstar,         ystar)
        r2 = np.hypot(xstar - S[j],  ystar)

        if r1 < 1e-10 or r2 < 1e-10:
            continue

        th1 = np.arctan2(ystar, xstar)
        th2 = np.arctan2(ystar, xstar - S[j])

        ln_r1r2  = np.log(r1 / r2)
        dth      = th1 - th2

        dphi = phi[i] - phi[j]

        # Katz & Plotkin eqs (11.26a, 11.26b)  — per unit source strength * 1/(2pi)
        CN[i, j] = (1.0/(2.0*np.pi)) * ( np.sin(dphi)*ln_r1r2 + np.cos(dphi)*dth )
        CT[i, j] = (1.0/(2.0*np.pi)) * ( np.cos(dphi)*ln_r1r2 - np.sin(dphi)*dth )

# =============================================================================
# BUILD SYSTEM  A*q = RHS
# Boundary condition: zero normal velocity at each control point
#   sum_j CN[i,j]*q[j]  +  V_inf·n_i  =  0
# =============================================================================
A   = CN.copy()
np.fill_diagonal(A, 0.5)          # self-influence = 0.5  (= pi/(2*pi))

# Freestream normal component at each panel
Vn_inf = Ux * nx + Uy * ny        # V_inf · n_i

RHS = -Vn_inf                     # enforce zero total normal velocity

# =============================================================================
# SOLVE FOR SOURCE STRENGTHS
# =============================================================================
q = np.linalg.solve(A, RHS)

print(f"Sum of source strengths (should be ~0): {np.sum(q * S):.6f}")

# =============================================================================
# TANGENTIAL VELOCITY  (surface velocity = what gives Cp)
# Vt[i] = freestream tangential + sum of source tangential influences
# =============================================================================
# Freestream tangential component
Vt_inf = Ux * np.cos(phi) + Uy * np.sin(phi)   # V_inf · t_i  shape (numPan,)

# Source-induced tangential: CT @ q
Vt_source = CT @ q                               # matrix multiply, shape (numPan,)

Vt = Vt_inf + Vt_source

# =============================================================================
# PRESSURE COEFFICIENT
# =============================================================================
Cp = 1.0 - (Vt / Vinf)**2

# =============================================================================
# UPPER / LOWER SPLIT
# CCW ordering: panels 0..LE_idx are lower surface, LE_idx..end are upper
# Find LE as panel with minimum XC
# =============================================================================
LE_idx = np.argmin(XC)
print(f"LE panel index: {LE_idx}  XC={XC[LE_idx]:.4f}  YC={YC[LE_idx]:.4f}")

# Lower surface: index 0 → LE_idx  (sort by increasing X)
# Upper surface: index LE_idx → numPan (sort by increasing X)
lower_idx = np.arange(0, LE_idx + 1)
upper_idx = np.arange(LE_idx, numPan)

# Sort both by X for clean plotting
lower_idx = lower_idx[np.argsort(XC[lower_idx])]
upper_idx = upper_idx[np.argsort(XC[upper_idx])]

# Exclude TE panels (X > 0.995) to remove spike
te_mask_l = XC[lower_idx] < 0.995
te_mask_u = XC[upper_idx] < 0.995
lower_idx = lower_idx[te_mask_l]
upper_idx = upper_idx[te_mask_u]

# =============================================================================
# PLOT
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(XC[upper_idx], Cp[upper_idx], 'b.-', label='Upper Surface')
plt.plot(XC[lower_idx], Cp[lower_idx], 'r.-', label='Lower Surface')
plt.xlabel('X/C')
plt.ylabel('Cp')
plt.title(f'Pressure Coefficient – NACA 0012 – AoA = {AoA}°')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# LIFT COEFFICIENT
# Cl = -sum( Cp * ny * S )  where ny is outward normal y-component
# =============================================================================
Cx = -np.sum(Cp * nx * S)  #same as Cy = -(Cp[0]*ny[0]*S[0] + Cp[1]*ny[1]*S[1] + ... + Cp[N]*ny[N]*S[N]) because the arrays are the same length
Cy = -np.sum(Cp * ny * S)

Cl = Cy*np.cos(AoA_r) - Cx*np.sin(AoA_r)
Cd = Cx*np.cos(AoA_r) + Cy*np.sin(AoA_r)
print(f"\nLift coefficient       Cl ≈ {Cl:.4f}")
print(f"Thin-airfoil theory:   Cl ≈ {2*np.pi*np.sin(AoA_r):.4f}")
