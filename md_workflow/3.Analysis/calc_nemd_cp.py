#!/usr/bin/env python3
# Usage: python update_cp_csv.py PID
#
# Updates:
#   ../../RESULTS/CP_MD_NEMD.csv   (PID, SMILES, CP)
#
# Reads (if present):
#   <PID>_Cp_EMD_result.dat

import sys
import csv
import re
from pathlib import Path

# --- Paths ---
SMILES_CSV = Path("../../..") / "SMILES.csv"
OUT_DIR = Path("../../RESULTS")
CP_CSV = OUT_DIR / "CP_MD_NEMD.csv"

PID_COL, SMILES_COL = "PID", "SMILES"
CP_COL = "CP"

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def usage():
    print("Usage: python update_cp_csv.py PID")
    sys.exit(1)


def first_number(text: str):
    m = FLOAT_RE.search(text)
    return m.group(0) if m else None


def load_smiles(pid: str) -> str:
    if not SMILES_CSV.exists():
        return ""
    with SMILES_CSV.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            return ""
        lower = {k.lower(): k for k in rdr.fieldnames}
        pid_k, smi_k = lower.get("pid"), lower.get("smiles")
        if not pid_k or not smi_k:
            return ""
        for row in rdr:
            if (row.get(pid_k) or "").strip() == pid:
                return (row.get(smi_k) or "").strip()
    return ""


def read_scalar(path: Path):
    if not path.exists():
        return None
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    s = first_number(txt)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def upsert_cp(csv_path: Path, pid: str, smiles: str, value_str: str):
    data = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and {PID_COL, SMILES_COL, CP_COL}.issubset(rdr.fieldnames):
                for row in rdr:
                    k = (row.get(PID_COL) or "").strip()
                    if k:
                        data[k] = {
                            PID_COL: k,
                            SMILES_COL: (row.get(SMILES_COL) or "").strip(),
                            CP_COL: (row.get(CP_COL) or "").strip(),
                        }

    data[pid] = {PID_COL: pid, SMILES_COL: smiles, CP_COL: value_str}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[PID_COL, SMILES_COL, CP_COL])
        w.writeheader()
        for v in data.values():
            w.writerow(v)


def main():
    if len(sys.argv) != 2:
        usage()

    pid = sys.argv[1].strip()
    if not pid:
        usage()

    cp_file = Path(f"{pid}_Cp_NEMD_result.dat")
    cp_val = read_scalar(cp_file)

    if cp_val is None:
        print(f"[SKIP] {cp_file.name}: not found or unreadable")
        print("[DONE] Nothing updated.")
        return

    smiles = load_smiles(pid)

    upsert_cp(CP_CSV, pid, smiles, f"{cp_val:.8f}")
    print(f"[OK] {cp_file.name} → {CP_CSV.name}: {CP_COL} = {cp_val:.8f}")
    print("[DONE] Updates complete.")


if __name__ == "__main__":
    main()
