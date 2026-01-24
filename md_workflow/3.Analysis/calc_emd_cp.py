#!/usr/bin/env python3
import os
import sys
import numpy as np


def compute_cp_emd(filename: str, n_windows: int = 8) -> float:
    """
    Compute Cp from an NPT trajectory (EMD/fluctuation method) using window-averaging.

    Input file columns (as produced by your LAMMPS post-processing):
        T[K], E_tot[kcal/mol], (pressure ignored), V[Å^3], rho[g/cm^3]

    Per window w:
        H(t)  = E(t) + P*V(t)                         [J]
        m_w   = <rho*V>_w                              [kg]
        Cp_w  = Var(H)_w / (kB * <T>_w^2 * m_w)        [J/(kg·K)]

    Uses P = 1 bar = 1.0e5 Pa.
    Returns:
        Cp_mean = mean(Cp_w over windows)              [J/(kg·K)]
    """

    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(-1, 5)

    T_all = data[:, 0]   # K
    E_kcalmol_all = data[:, 1]  # kcal/mol
    V_ang3_all = data[:, 3]     # Å^3
    rho_gcc_all = data[:, 4]    # g/cm^3

    n_frames = len(data)
    if n_frames < n_windows:
        raise ValueError(f"Not enough frames ({n_frames}) for {n_windows} windows.")

    # Constants
    ANG3_TO_M3 = 1e-30
    KCAL_TO_J = 4184.0
    NA = 6.02214076e23
    KB = 1.380649e-23      # J/K
    P_CONST_PA = 1.0e5     # 1 bar in Pa

    # Per-frame conversions
    V_m3_all = V_ang3_all * ANG3_TO_M3
    rho_kgm3_all = rho_gcc_all * 1000.0
    m_kg_all = rho_kgm3_all * V_m3_all                      # kg
    E_J_all = E_kcalmol_all * KCAL_TO_J / NA                # J
    H_J_all = E_J_all + P_CONST_PA * V_m3_all               # J

    # Split into equal windows
    window_size = n_frames // n_windows
    n_used = window_size * n_windows
    if n_used != n_frames:
        print(f"[WARN] Using first {n_used} of {n_frames} frames to form {n_windows} equal windows.")

    T_all = T_all[:n_used]
    H_all = H_J_all[:n_used]
    m_all = m_kg_all[:n_used]

    # Window-wise Cp
    Cp_list = []
    for w in range(n_windows):
        i0 = w * window_size
        i1 = (w + 1) * window_size

        Tw = float(np.mean(T_all[i0:i1]))
        Hw = H_all[i0:i1]
        mw = float(np.mean(m_all[i0:i1]))

        var_Hw = float(np.var(Hw, ddof=1))  # unbiased variance
        Cp_w = var_Hw / (KB * (Tw ** 2) * mw)  # J/(kg·K)
        Cp_list.append(Cp_w)

    return float(np.mean(np.array(Cp_list)))


def main():
    """
    Usage:
        python emd_cp.py <pid> [n_windows]

    Expects: <pid>_npt_properties.dat in cwd.
    Outputs:
        <pid>_Cp_EMD_result.dat  (single value: Cp_mean in J/(kg·K))
    """
    if len(sys.argv) < 2:
        print("Usage: python emd_cp.py <pid> [n_windows]")
        sys.exit(1)

    pid = sys.argv[1]
    n_windows = 8

    # Optional integer n_windows
    for arg in sys.argv[2:]:
        if arg.strip().isdigit():
            n_windows = int(arg)

    infile = os.path.join(os.getcwd(), f"{pid}_npt_properties.dat")
    if not os.path.isfile(infile):
        print(f"Error: input file not found: {infile}")
        sys.exit(1)

    try:
        Cp_mean = compute_cp_emd(infile, n_windows=n_windows)

        print(f"PID       : {pid}")
        print(f"n_windows : {n_windows}")
        print(f"Cp_mean   : {Cp_mean:.8f} J/(kg·K)")

        np.savetxt(f"{pid}_Cp_EMD_result.dat", [Cp_mean], fmt="%.8f")
        print("\n[DONE] Saved Cp mean value.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
