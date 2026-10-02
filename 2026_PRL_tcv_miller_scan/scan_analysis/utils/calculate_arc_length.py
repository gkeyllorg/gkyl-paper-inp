import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def calculate_arc_length(r, kappa, delta, q, theta_min = 0, theta_max=2*np.pi, R0=1.0, n_points=200):
    """
    Computes the distance along a magnetic field line l(theta).
    Assumes straight field line coordinates where dphi/dtheta = q.
    """
    # 1. Define Grid
    theta = np.linspace(theta_min, theta_max, n_points)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    alpha = np.arcsin(delta)

    # 2. Coordinate Functions & Derivatives
    # Argument for the cosine term in R
    angle_arg = theta + alpha * sin_theta
    
    # R and Z values
    R = R0 + r * np.cos(angle_arg)
    # Z = r * kappa * sin_theta (Not strictly needed for dl, only derivatives are)

    # Derivatives with respect to theta
    # dR/dtheta = -r * sin(u) * u'
    dR_dtheta = -r * np.sin(angle_arg) * (1 + alpha * cos_theta)
    
    # dZ/dtheta = r * kappa * cos(theta)
    dZ_dtheta = r * kappa * cos_theta

    # 3. Construct Metric
    # Poloidal contribution: dl_p^2 = R'^2 + Z'^2
    dl_poloidal_sq = dR_dtheta**2 + dZ_dtheta**2
    
    # Toroidal contribution: dl_phi^2 = (R * dphi/dtheta)^2 approx (R * q)^2
    dl_toroidal_sq = (R * q)**2
    
    # Total differential element
    dl_dtheta = np.sqrt(dl_poloidal_sq + dl_toroidal_sq)

    # 4. Integrate to get Arc Length l
    l = cumulative_trapezoid(dl_dtheta, theta, initial=0)

    return theta, l


import numpy as np
from scipy.optimize import newton

def find_bounce_angle(Rb, r, delta, R0=1.0):
    # 1. Safety Check: Ensure Rb is reachable
    R_min = R0 - r
    R_max = R0 + r
    if not (R_min <= Rb <= R_max):
        raise ValueError(f"Bounce radius {Rb} is outside the flux surface [{R_min}, {R_max}]")

    # 2. Calculate the target argument value
    # We use abs() to handle the symmetry (bounce happens at +/- theta)
    cos_arg = (Rb - R0) / r
    xi = np.arccos(cos_arg)  # Returns value in [0, pi]
    
    alpha = np.arcsin(delta)

    # 3. Define the function to zero: f(theta) = theta + alpha*sin(theta) - xi
    func = lambda theta: theta + alpha * np.sin(theta) - xi
    
    # derivative f'(theta) = 1 + alpha*cos(theta) (helps convergence)
    fprime = lambda theta: 1 + alpha * np.cos(theta)

    # 4. Solve
    # Initial guess: xi / (1 + delta) is a good start
    theta_b = newton(func, x0=xi/(1+delta), fprime=fprime)
    
    return theta_b # Result in radians

import numpy as np
from scipy.optimize import newton

def get_theta_from_R(R_target, r, delta, R0=1.0, tolerance=1e-6):
    """
    Inverts the Miller parameterization R(theta) to find theta given R.
    
    Equation: R = R0 + r * cos(theta + arcsin(delta) * sin(theta))
    
    Args:
        R_target (float): The major radius coordinate to solve for.
        r (float): Minor radius of the flux surface.
        delta (float): Triangularity.
        R0 (float): Geometric center major radius.
        tolerance (float): Convergence threshold for the solver.
        
    Returns:
        float: Poloidal angle theta (in radians) within [0, pi].
    """
    # 1. Validate Inputs
    R_min = R0 - r
    R_max = R0 + r
    
    if not (R_min - tolerance <= R_target <= R_max + tolerance):
        raise ValueError(f"Target R={R_target} is outside flux surface range [{R_min:.3f}, {R_max:.3f}]")
    
    # Clamp value to handle floating point noise near edges
    cos_val = (R_target - R0) / r
    cos_val = np.clip(cos_val, -1.0, 1.0)
    
    # 2. Compute the Target Argument (xi)
    # The Miller R equation is: cos(theta + alpha*sin(theta)) = (R - R0)/r
    # Let the argument inside cosine be xi.
    xi = np.arccos(cos_val) # Returns value in [0, pi]
    
    alpha = np.arcsin(delta)

    # 3. Define the Transcendental Equation
    # We need to solve: f(theta) = theta + alpha*sin(theta) - xi = 0
    def f(theta):
        return theta + alpha * np.sin(theta) - xi
    
    def f_prime(theta):
        return 1 + alpha * np.cos(theta)

    # 4. Solve for Theta
    # Initial guess: For small delta, theta approx xi. 
    # Dividing by (1+delta) accounts for the compression at the top.
    guess = xi / (1 + delta/2)
    
    theta_solution = newton(f, x0=guess, fprime=f_prime, tol=tolerance)
    
    return theta_solution
