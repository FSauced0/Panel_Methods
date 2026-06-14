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


# Panel outward normal angle (90 deg CCW from tangent, pointing away from body)
delta = phi + 0.5 * np.pi

# =============================================================================
# BUILD INFLUENCE COEFFICIENT MATRIX  A(i,j)
# Normal velocity at control point i due to unit source strength on panel j
# =============================================================================
AV= np.zeros((numPan,numPan))
A_ext = np.zeros((numPan+1, numPan+1))



for i in range(numPan):
    for j in range(numPan):
        if i == j:
            # Self-influence: normal velocity = 0.5  (= pi / 2pi)
            AV[i, j] = 0.5

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
            AV[i,j]= (-dtheta*dot_ustar+ log_r*dot_vstar)
A_ext[:numPan, :numPan] = AV
for i in range(numPan):
    A_ext[i, numPan] = -0.5
RHS_ext = np.zeros(numPan+1)
#
# =============================================================================
# RIGHT-HAND SIDE: negative freestream normal component at each control point
# =============================================================================

for i in range(numPan):
    # Panel i outward normal direction: (sin(phi[i]), -cos(phi[i]))
    RHS_ext[i] = -(Ux * (-np.sin(phi[i])) + Uy * np.cos(phi[i]))
    



#________________________________________________________________________

AT= np.zeros((numPan,numPan))
Vt_free = np.zeros(numPan)

for i in range(numPan):
    # Freestream tangential component on panel i
    Vt_free[i] = Ux * np.cos(phi[i]) + Uy * np.sin(phi[i])

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

        AT[i,j] = (log_r * dot_ustar_t + dtheta * dot_vstar_t) / (2.0 * np.pi)
# --- fill Kutta row ---
A_ext[numPan, :numPan] = AT[0, :] + AT[numPan-1, :]
A_ext[numPan, numPan]  = 1.0
RHS_ext[numPan]        = -(Vt_free[0] + Vt_free[numPan-1])

# --- solve ---
solution = np.linalg.solve(A_ext, RHS_ext)
q   = solution[:numPan]
gamma = solution[numPan]

# --- now compute Vt using q and gamma ---
Vt = Vt_free + AT @ q + 0.5 * gamma

Cp = 1 - (Vt / Vinf)**2
Cl = 2 * gamma * np.sum(S) / Vinf

plt.figure()
plt.plot(XC, Cp, 'b.-')
plt.gca().invert_yaxis()  # Cp convention: negative up
plt.xlabel('X')
plt.ylabel('Cp')
plt.title('Pressure Coefficient')
plt.grid(True)
plt.show()
