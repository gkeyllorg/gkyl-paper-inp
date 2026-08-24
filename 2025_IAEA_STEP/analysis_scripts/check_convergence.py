import numpy as np
from scipy.stats import linregress

def check_gkeyll_convergence(time, signal, window=1000, 
                            slope_threshold=0.016, 
                            z_threshold=2.0, 
                            cv_threshold=0.05,
                            cohen_threshold=0.2):
    """
    Analyzes a SOLPS time trace (e.g., T_sep, n_sep) for steady-state convergence
    using three criteria: Linear Drift, Two-Window Mean consistency, and CV (Noise).
    
    Args:
        time (array): Simulation time or step array.
        signal (array): The data trace (e.g., T_sep at midplane).
        window (int): Number of recent steps to analyze.
        slope_threshold (float): Max allowed relative drift (0.016 = 1.6%).
        z_threshold (float): Max Z-score allowed between the two window halves.
        cv_threshold (float): Max allowed noise-to-signal ratio (CV).
        
    Returns:
        dict: A summary of metrics and booleans for pass/fail.
    """
    
    # Safety check for window size
    if len(signal) < window:
        print(f"Warning: Signal length ({len(signal)}) is shorter than window ({window}). Using full length.")
        window = len(signal)

    # 1. Extract the analysis window
    t_win = time[-window:]
    y_win = signal[-window:]
    
    # Calculate global stats for window
    y_mean_total = np.mean(y_win)
    y_std_total = np.std(y_win, ddof=1)

    # --- TEST 1: LINEAR DRIFT (The Slope Check) ---
    slope, intercept, r_val, p_val, std_err = linregress(t_win, y_win)
    
    # Total change predicted by slope over the window duration
    total_drift_abs = abs(slope * (t_win[-1] - t_win[0]))
    relative_drift = total_drift_abs / abs(y_mean_total)
    
    drift_pass = relative_drift < slope_threshold

    # --- TEST 2: TWO-WINDOW TEST (The Mean Consistency Check) ---
    midpoint = len(y_win) // 2
    y_w1 = y_win[:midpoint]  # "Older" half
    y_w2 = y_win[midpoint:]  # "Newer" half
    
    mu1, std1 = np.mean(y_w1), np.std(y_w1, ddof=1)
    mu2, std2 = np.mean(y_w2), np.std(y_w2, ddof=1)
    n1, n2 = len(y_w1), len(y_w2)
    
    # Calculate Z-score
    variance_sum = (std1**2 / n1) + (std2**2 / n2)
    if variance_sum == 0:
        z_score = 0.0
    else:
        z_score = abs(mu1 - mu2) / np.sqrt(variance_sum)

    # Cohen's d (Physical Effect Size)
    # Measures difference relative to noise, independent of N
    s_pooled = np.sqrt((std1**2 + std2**2) / 2)
    cohen_d = abs(mu1 - mu2) / s_pooled if s_pooled > 0 else 0
        
    # Override: if the relative difference is physically negligible (< 0.5%), pass anyway
    rel_diff_means = abs(mu1 - mu2) / abs(y_mean_total)
    #window_pass = (z_score < z_threshold) or (rel_diff_means < 0.005)
    # Alternative to Override: PASS if Z-score is low OR if the effect size is trivial
    window_pass = (z_score < z_threshold) or (cohen_d < cohen_threshold) or (rel_diff_means < 0.005)

    # --- TEST 3: COEFFICIENT OF VARIATION (The Noise Check) ---
    # CV = Standard Deviation / Mean
    cv = y_std_total / abs(y_mean_total)
    
    # Note: High CV doesn't mean it's drifting, but it means the "Steady State" 
    # is too noisy to trust without averaging over a massive window.
    cv_pass = cv < cv_threshold

    # --- FINAL VERDICT ---
    # It must pass drift AND window consistency. 
    # CV is usually a "Quality Warning" rather than a hard stop, but strict mode requires it.
    is_converged = drift_pass and window_pass and cv_pass

    # Print Report
    print(f"--- Convergence Report for Last {window} Steps ---")
    print(f"Mean Value:      {y_mean_total:.4e}")
    print(f"[Test 1] Drift:  {relative_drift:.4%} (Limit: {slope_threshold:.1%}) -> {'PASS' if drift_pass else 'FAIL'}")
    print(f"[Test 2] Z-Score:{z_score:.4f}     (Limit: {z_threshold})   -> {'PASS' if window_pass else 'FAIL'}")
    print(f"         Rel Diff: {rel_diff_means:.4%}")
    print(f"         Cohen's d: {cohen_d:.4f}   -> {'PASS (Trivial Diff)' if cohen_d < cohen_threshold else 'SIGNIFICANT'}")
    print(f"[Test 3] CV (Noise): {cv:.4%} (Limit: {cv_threshold:.1%}) -> {'PASS' if cv_pass else 'FAIL'}")
    
    status = "STEADY STATE" if is_converged else "NOT CONVERGED"
    print(f"FINAL RESULT:    {status}")
    print("-" * 45)

    return {
        "converged": is_converged,
        "drift_metric": relative_drift,
        "z_score": z_score,
        "cv": cv,
        "mean": y_mean_total
    }

# --- Example Usage ---
#if __name__ == "__main__":
#    # Create dummy data
#    steps = np.arange(1000)
#    # A steady signal (100) with 2% noise (sigma=2)
#    signal = 100 + np.random.normal(0, 2, size=1000)
#    
#    # Run Check
#    check_solps_convergence(steps, signal, cv_threshold=0.05)
