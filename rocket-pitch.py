from scipy.linalg import solve_continuous_are
import numpy as np
import matplotlib.pyplot as plt


# Owner's own example: single-axis rocket PITCH attitude control via thrust
# vector control (TVC).  Control-affine, same state dimension for model and
# plant (Assumption 1).  Tier 1 = AERO NONLINEARITY ONLY (cubic Duffing-like
# pitching moment).  Fuel slosh and mass burn are LATER tiers -- design is built
# to extend to them, but they are not implemented here.
#
# This is the paper's Example 1 (perturbed oscillator) generalized to a rocket:
# a finless / aerodynamically-unstable airframe stabilized by gimballing.



# State / control
#   x = [theta, theta_dot]   theta = pitch angle [rad], theta_dot = pitch rate
#   u = delta                gimbal deflection [rad] (scalar control)
#   xdot = f(x) + g(x)*u     control-affine


# Physical parameter mapping (documented; we use clean rounded values below):
#   a1 = q*S*d*Cm_alpha / I     linear aero pitching-moment slope / inertia
#   a3 = q*S*d*Cm_3    / I      CUBIC aero moment (Duffing-like) / inertia
#   b  = T * ell / I            TVC control authority: thrust * moment arm / I
# Finless rocket => Cm_alpha > 0 => a1 > 0 => statically UNSTABLE in pitch
# (the airframe pitches away from the wind), stabilized only by TVC.
#
# Clean Tier-1 values (rad, s):
a1 = 4.0    # unstable: open-loop poles at +/- sqrt(a1) = +/- 2 rad/s
b  = 8.0    # control authority (T*ell/I); g is CONSTANT here

# Cubic-aero strength is the SCENARIO KNOB (analogous to rho in section 7.2).
# Same A_m / same K in both scenarios -- only the perturbation a3 changes.
a3_mild   = 2.0     # weak cubic  -> mismatch stays under threshold, NO switch
a3_strong = 60.0    # strong cubic -> mismatch grows -> switch fires, then RL


def f(x, a3):
    # true nonlinear drift: linear aero + cubic Duffing-like aero moment
    theta, theta_dot = x
    return np.array([theta_dot,
                     a1 * theta + a3 * theta**3])


def g(x):
    # control vector field: constant, gimbal torque enters the rate equation
    return np.array([0.0, b])



# Available (deliberately-imperfect) model: LINEAR AERO ONLY.
# Same state dimension (Assumption 1); the only thing the model is missing is
# the cubic a3*theta^3.  Because g(x) is constant, B_m = g EXACTLY -- there is
# ZERO input-channel mismatch; ALL model error is the omitted cubic in f.

Am = np.array([[0.0, 1.0],
               [a1,  0.0]])
Bm = np.array([[0.0],
               [b]])

# LQR weights.  Penalize pitch angle (regulate attitude); modest rate / control.
Q = np.diag([1.0, 0.1])
R = np.array([[0.2]])

# Model-based LQR gain (Eq. 11): u_m = -K x, from the inaccurate (linear) model.
Pm = solve_continuous_are(Am, Bm, Q, R)
K  = np.linalg.inv(R) @ Bm.T @ Pm     # shape (1, 2)


# Near-origin ground-truth augmentation gain K_tilde_lin (for later validation).
# The ONLY nonlinearity is a3*theta^3, whose Jacobian at the origin is zero, so
# the linearized true plant EQUALS A_m exactly.  Hence K_tilde_lin is just the
# section-7.1 augmented-Riccati gain with A -> A_m:
#   K_tilde = inv(R) B^T ARE(A_m - B_m K, B_m, Q, R)
# The RL augmentation ũ(x) should match -K_tilde_lin x for SMALL states (where
# the cubic is negligible).  Computed now; used in step 5.
P_tilde = solve_continuous_are(Am - Bm @ K, Bm, Q, R)
K_tilde_lin = np.linalg.inv(R) @ Bm.T @ P_tilde   # shape (1, 2)


# STEP-1 CHECKS: open-loop instability + controllability of (A_m, B_m).
# These must actually be verified, not just asserted.

eig_open = np.linalg.eigvals(Am)
ctrb = np.hstack([Bm, Am @ Bm])           # [B_m, A_m B_m]
ctrb_rank = np.linalg.matrix_rank(ctrb)
ctrb_det = np.linalg.det(ctrb)

# closed-loop poles under the model-based gain (sanity: should be stable)
eig_closed = np.linalg.eigvals(Am - Bm @ K)

if __name__ == "__main__":
    print("=== Step 1: dynamics, model, LQR gain, structural checks ===")
    print(f"a1 = {a1},  b = {b}")
    print(f"open-loop eig(A_m)        = {eig_open}   (unstable iff any Re > 0)")
    print(f"controllability [B, A B]  =\n{ctrb}")
    print(f"  rank = {ctrb_rank} (need 2),  det = {ctrb_det:.3f} (need != 0)")
    print(f"model-based gain K        = {K}")
    print(f"closed-loop eig(A_m-B_mK) = {eig_closed}   (should be stable)")
    print(f"K_tilde_lin (val target)  = {K_tilde_lin}")
