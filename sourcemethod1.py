#PANEL METHOD AIRFOIL SOLVER
#Written by: Fabian Saucedo

#Status: Does work 

#Notes: 
#THIS IS A SOURCE PANEL METHOD ONLY
#Works only for non-lifting bodies or symmeterical airfoils at 0AOA
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# LOAD GEOMETRY
# =============================================================================
df = pd.read_csv(r'C:\Users\fabia\Downloads\v23010.dat', sep=r'\s+', header=None, skiprows=1)

XB = df.iloc[:, 0].values
YB = df.iloc[:, 1].values

print(f"First point: ({XB[0]:.4f}, {YB[0]:.4f})")
print(f"Last point:  ({XB[-1]:.4f}, {YB[-1]:.4f})")
print(f"Total points: {len(XB)}")

# =============================================================================
# VISUALIZE AIRFOIL SHAPE
# =============================================================================
plt.figure()
plt.plot(XB, YB, 'b.-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Airfoil Shape')
plt.axis('equal')
plt.grid(True)
plt.show()

# =============================================================================
# ENFORCE CLOCKWISE (CW) PANEL ORDERING
# Using the shoelace/signed-area method: positive sum = CW
# =============================================================================
numPan = len(XB) - 1

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

print(f"First point after check: ({XB[0]:.4f}, {YB[0]:.4f})")
print(f"Last point after check:  ({XB[-1]:.4f}, {YB[-1]:.4f})")

# =============================================================================
# ANGLE OF ATTACK
# =============================================================================
AoA    = 0                        # degrees
AoA_r  = AoA * np.pi / 180.0       # radians

# Freestream unit vector components
Vinf  = 1.0
Ux    = Vinf * np.cos(AoA_r)
Uy    = Vinf * np.sin(AoA_r)
# =============================================================================
# PANEL GEOMETRY: control points, lengths, and angles
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

# Panel outward normal angle (90 deg CCW from tangent, pointing away from body)
delta = phi + 0.5 * np.pi

# =============================================================================
# BUILD INFLUENCE COEFFICIENT MATRIX  A(i,j)
# Normal velocity at control point i due to unit source strength on panel j
# =============================================================================
A = np.zeros((numPan, numPan))

for i in range(numPan):
    for j in range(numPan):

        if i == j:
            # Self-influence: normal velocity = 0.5  (= pi / 2pi)
            A[i, j] = 0.5

        else:
            # --- translate control point i into panel j local frame ---
            xt =  XC[i] - XB[j]
            yt =  YC[i] - YB[j]

            xstar =  xt * np.cos(phi[j]) + yt * np.sin(phi[j])   # along panel j
            ystar = -xt * np.sin(phi[j]) + yt * np.cos(phi[j])   # normal to panel j

            # Distances to panel j endpoints
            r1 = np.sqrt(xstar**2         + ystar**2)
            r2 = np.sqrt((xstar - S[j])**2 + ystar**2)

            # Angles to panel j endpoints (avoid log(0))
            th1 = np.arctan2(ystar, xstar)
            th2 = np.arctan2(ystar, xstar - S[j])

            # Velocity components in panel j local frame
            # u* (along  panel j) = log(r1/r2) / (2*pi)
            # v* (normal panel j) = (th1 - th2)  / (2*pi)
            log_r  = np.log(r1 / r2) / (2.0 * np.pi)   if r1 > 1e-10 and r2 > 1e-10 else 0.0
            dtheta = (th1 - th2)     / (2.0 * np.pi)

            # Project onto panel i outward normal
            # panel j local x-axis in global: ( cos(phi[j]),  sin(phi[j]) )
            # panel j local y-axis in global: (-sin(phi[j]),  cos(phi[j]) )
            # panel i outward normal in global: (-sin(phi[i]), cos(phi[i]) )
            dot_ustar = np.sin(phi[i] - phi[j])          # cos(phi[j])*(-sin(phi[i])) + sin(phi[j])*(cos(phi[i]))  -- simplified
            dot_vstar = np.cos(phi[i] - phi[j])          # -sin(phi[j])*(-sin(phi[i])) + cos(phi[j])*(cos(phi[i])) -- simplified

            A[i, j] = log_r * dot_ustar + dtheta * dot_vstar

            # Guard against numerical garbage
            if not np.isfinite(A[i, j]):
                A[i, j] = 0.0

# =============================================================================
# RIGHT-HAND SIDE: negative freestream normal component at each control point
# =============================================================================
RHS = np.zeros(numPan)
for i in range(numPan):
    # Panel i outward normal direction: (-sin(phi[i]), cos(phi[i]))
    RHS[i] = -(Ux * (np.sin(phi[i])) + Uy * -np.cos(phi[i]))
    

# =============================================================================
# SOLVE FOR SOURCE STRENGTHS  q
# =============================================================================
q = np.linalg.solve(A, RHS)

print(f"\nSum of source strengths (should be ~0): {np.sum(q * S):.6f}")

# =============================================================================
# COMPUTE TANGENTIAL (surface) VELOCITY AT EACH CONTROL POINT
# =============================================================================
Vt = np.zeros(numPan)

for i in range(numPan):
    # Freestream tangential component on panel i
    Vt[i] = Ux * np.cos(phi[i]) + Uy * np.sin(phi[i])

    for j in range(numPan):
        if i == j:
            # Self-tangential contribution of a constant-source panel = 0
            continue

        xt    =  XC[i] - XB[j]
        yt    =  YC[i] - YB[j]
        xstar =  xt * np.cos(phi[j]) + yt * np.sin(phi[j])
        ystar = -xt * np.sin(phi[j]) + yt * np.cos(phi[j])

        r1  = np.sqrt(xstar**2          + ystar**2)
        r2  = np.sqrt((xstar - S[j])**2 + ystar**2)
        th1 = np.arctan2(ystar, xstar)
        th2 = np.arctan2(ystar, xstar - S[j])

        log_r  = np.log(r1 / r2) if r1 > 1e-10 and r2 > 1e-10 else 0.0
        dtheta = th1 - th2

        # Project panel j local velocities onto panel i tangent direction
        # Panel i tangent in global: (cos(phi[i]), sin(phi[i]))
        # dot with panel j local x-axis: cos(phi[i]-phi[j])
        # dot with panel j local y-axis: -sin(phi[i]-phi[j])   (note sign)
        dot_ustar_t =  np.cos(phi[i] - phi[j])
        dot_vstar_t = -np.sin(phi[i] - phi[j])

        contrib = (q[j] / (2.0 * np.pi)) * (log_r * dot_ustar_t + dtheta * dot_vstar_t)

        if np.isfinite(contrib):
            Vt[i] += contrib

# =============================================================================
# PRESSURE COEFFICIENT  Cp = 1 - (Vt/V_inf)^2   (V_inf = 1)
# =============================================================================
Cp = 1.0 - Vt**2

# =============================================================================
# SPLIT UPPER / LOWER SURFACE BY Y-COORDINATE OF CONTROL POINT
# =============================================================================
upper_mask = YC >= 0.0
lower_mask = YC <  0.0

# Sort each surface by X so the plot lines go left → right
upper_idx = np.where(upper_mask)[0][np.argsort(XC[upper_mask])]
lower_idx = np.where(lower_mask)[0][np.argsort(XC[lower_mask])]

# =============================================================================
# PLOT PRESSURE COEFFICIENT
# =============================================================================
plt.figure(figsize=(9, 5))
plt.plot(XC[upper_idx], Cp[upper_idx], 'b.-', label='Upper Surface')
plt.plot(XC[lower_idx], Cp[lower_idx], 'r.-', label='Lower Surface')
plt.xlabel('X/C')
plt.ylabel('Cp')
plt.title(f'Pressure Coefficient  –  AoA = {AoA}°')
plt.gca().invert_yaxis()      # convention: suction (negative Cp) plotted upward
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# LIFT COEFFICIENT  (pressure integration)
# Cl = -∮ Cp · n_y · ds   (n_y = cos(phi_i) for outward normal)
# For a unit-chord, V_inf = 1 airfoil
# =============================================================================
Cl = -np.sum(Cp * np.cos(delta) * S)          # delta = outward normal angle
print(f"\nLift coefficient Cl ≈ {Cl:.4f}")
print(f"(Thin-airfoil theory prediction: {2*np.pi*np.sin(AoA_r):.4f})")
