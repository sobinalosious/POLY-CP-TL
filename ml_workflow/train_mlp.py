#!/usr/bin/env python3
"""
train_cp_nestedcv_mlp_fpdesc.py

Nested-CV + (optional) low-fidelity pretraining + transfer learning using a PyTorch MLP.

Core ML workflow:
  1) EXP-only nested CV (outer CV test; inner CV Optuna tuning)
  2) LOW-fidelity pretraining (arch + train HP tuning on LOW)
  3) Transfer learning: fine-tune pretrained model on EXP-train
     - Architecture fixed from pretraining; fine-tuning tunes training HPs only.

Featurization options:
  - fp_method: morgan, rdkit, maccs, topologicaltorsion, atompair, pe
  - optional 7 RDKit descriptors via --use_descriptors

IMPORTANT HPC FIXES:
  - RDKit BitVect conversion: avoid DataStructs.ConvertToNumpyArray entirely
    (some HPC RDKit builds reject numpy arrays). We use ToBitString() -> numpy.
  - PE (mol2vec) with gensim 4.x: avoid mol2vec.features.sentences2vec() (uses removed wv.vocab).
    We compute sentence embedding as mean of token vectors using wv.key_to_index.

Expected CSV columns:
  - SMILES column (default: SMILES)
  - target column (default: Cp)
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors


# -----------------------------
# Repro / IO helpers
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def mean_std_from_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if not fold_metrics:
        return None, None
    keys = list(fold_metrics[0].keys())
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return mean, std


def print_mean_std(label: str, fold_metrics: List[Dict[str, float]]):
    mean, std = mean_std_from_fold_metrics(fold_metrics)
    if mean is None or std is None:
        print(f"[{label}] No folds completed.")
        return
    print(f"\n[{label}] Mean ± Std over outer folds")
    print(f"  R2   : {mean['r2']:.4f} ± {std['r2']:.4f}")
    print(f"  RMSE : {mean['rmse']:.4f} ± {std['rmse']:.4f}")
    print(f"  MAE  : {mean['mae']:.4f} ± {std['mae']:.4f}\n")


# -----------------------------
# Plotting (matplotlib)
# -----------------------------
def save_parity(y_true, y_pred, title: str, outpath: str):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad

    m = metrics(y_true, y_pred)
    plt.figure(figsize=(6.8, 6.8))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolor="k", linewidths=0.5)
    plt.plot([lo, hi], [lo, hi], "--k", linewidth=1.5)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.text(
        0.05, 0.95,
        f"R²={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nMAE={m['mae']:.3f}",
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.4", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# Feature engineering
# -----------------------------
def _mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def smiles_to_descriptors(smiles_list: List[str]) -> np.ndarray:
    """
    7 descriptors:
      MolWt, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, FractionCSP3
    """
    desc_fns = [
        Descriptors.MolWt,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.NumRotatableBonds,
        Descriptors.RingCount,
        Descriptors.FractionCSP3,
    ]
    out = np.full((len(smiles_list), len(desc_fns)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        m = _mol(smi)
        if m is None:
            continue
        out[i, :] = np.array([fn(m) for fn in desc_fns], dtype=np.float32)
    return out


def _bv_to_float_array_via_bitstring(bv, n_bits: int) -> np.ndarray:
    """
    HPC-safe BitVect -> numpy conversion.
    Avoids DataStructs.ConvertToNumpyArray which may fail due to RDKit/NumPy ABI issues on clusters.
    """
    s = bv.ToBitString()  # '0'/'1' string, length = n_bits
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    return arr.astype(np.float32, copy=False)


def smiles_to_fingerprint(
    smiles_list: List[str],
    fp_method: str,
    morgan_radius: int,
    n_bits: int,
    pe_model_path: Optional[str] = None,
) -> np.ndarray:
    """
    Fingerprint choices:
      - morgan, rdkit, maccs, topologicaltorsion, atompair (RDKit bit vectors)
      - pe (mol2vec Word2Vec): mean token embedding (gensim 4 compatible)
    """
    fp_method = fp_method.lower()

    if fp_method == "pe":
        if pe_model_path is None:
            raise ValueError("--pe_model_path is required when --fp_method pe")
        try:
            from gensim.models import word2vec
            from mol2vec.features import mol2alt_sentence, MolSentence
            from rdkit.Chem import rdBase
            rdBase.DisableLog("rdApp.error")
        except Exception as e:
            raise ImportError(
                "fp_method=pe requires mol2vec + gensim installed and importable."
            ) from e

        model = word2vec.Word2Vec.load(pe_model_path)
        wv = model.wv

        sentences = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            m = _mol(smi)
            if m is None:
                continue
            sentences.append(MolSentence(mol2alt_sentence(m, 1)))
            valid_indices.append(i)

        # gensim-4 compatible token lookup (no wv.vocab)
        unk = "UNK"
        unk_vec = wv[unk] if unk in wv.key_to_index else np.zeros((wv.vector_size,), dtype=np.float32)

        X = np.zeros((len(smiles_list), wv.vector_size), dtype=np.float32)
        for j, idx in enumerate(valid_indices):
            toks = list(sentences[j])
            vecs = []
            for t in toks:
                if t in wv.key_to_index:
                    vecs.append(wv[t])
                else:
                    vecs.append(unk_vec)
            X[idx, :] = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        return X

    fps: List[Optional[np.ndarray]] = []

    for smi in smiles_list:
        m = _mol(smi)
        if m is None:
            fps.append(None)
            continue

        if fp_method == "morgan":
            bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, morgan_radius, nBits=n_bits)
            fps.append(_bv_to_float_array_via_bitstring(bv, n_bits))

        elif fp_method == "rdkit":
            bv = Chem.RDKFingerprint(m, fpSize=n_bits, maxPath=morgan_radius)
            fps.append(_bv_to_float_array_via_bitstring(bv, n_bits))

        elif fp_method == "maccs":
            bv = MACCSkeys.GenMACCSKeys(m)  # 167 bits
            nb = bv.GetNumBits()
            fps.append(_bv_to_float_array_via_bitstring(bv, nb))

        elif fp_method == "topologicaltorsion":
            bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=n_bits)
            fps.append(_bv_to_float_array_via_bitstring(bv, n_bits))

        elif fp_method == "atompair":
            bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits)
            fps.append(_bv_to_float_array_via_bitstring(bv, n_bits))

        else:
            raise ValueError(f"Unsupported fp_method: {fp_method}")

    # Build (N, D) with NaNs for invalid SMILES
    d = None
    for v in fps:
        if v is not None:
            d = v.shape[0]
            break

    if d is None:
        return np.full((len(smiles_list), 1), np.nan, dtype=np.float32)

    X = np.full((len(smiles_list), d), np.nan, dtype=np.float32)
    for i, v in enumerate(fps):
        if v is None:
            continue
        X[i, :] = v
    return X


def featurize_smiles(
    smiles_list: List[str],
    fp_method: str,
    use_descriptors: bool,
    morgan_radius: int,
    n_bits: int,
    pe_model_path: Optional[str],
) -> np.ndarray:
    Xfp = smiles_to_fingerprint(
        smiles_list=smiles_list,
        fp_method=fp_method,
        morgan_radius=morgan_radius,
        n_bits=n_bits,
        pe_model_path=pe_model_path,
    )
    if not use_descriptors:
        return Xfp.astype(np.float32, copy=False)

    Xdesc = smiles_to_descriptors(smiles_list).astype(np.float32, copy=False)
    return np.concatenate([Xfp, Xdesc], axis=1).astype(np.float32, copy=False)


def filter_valid_rows(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    good = np.isfinite(y)
    good &= np.all(np.isfinite(X), axis=1)
    df2 = df.loc[good].reset_index(drop=True)
    return df2, X[good], y[good]


# -----------------------------
# MLP model
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z).squeeze(-1)

    def reset_head(self):
        nn.init.kaiming_uniform_(self.head.weight, a=np.sqrt(5))
        if self.head.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.head.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.head.bias, -bound, bound)


# -----------------------------
# Training utils
# -----------------------------
@dataclass
class TrainCfg:
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float = 0.0


def set_trainable(model: nn.Module, freeze_mode: str):
    """
    freeze_mode:
      - "none": train all layers
      - "backbone": freeze backbone, train head only
    """
    freeze_mode = freeze_mode.lower()
    if freeze_mode == "none":
        for p in model.parameters():
            p.requires_grad = True
    elif freeze_mode == "backbone":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")


def maybe_standardize_fit(X_train: np.ndarray, use: bool):
    if not use:
        return None, X_train
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (mu, sd), (X_train - mu) / sd


def maybe_standardize_apply(X: np.ndarray, scaler):
    if scaler is None:
        return X
    mu, sd = scaler
    return (X - mu) / sd


def train_one(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    device: torch.device,
    cfg: TrainCfg,
) -> nn.Module:
    model = model.to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.tensor(yva, dtype=torch.float32, device=device)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    bad = 0

    n = Xt.shape[0]
    idx = torch.arange(n, device=device)

    for _epoch in range(cfg.max_epochs):
        model.train()
        perm = idx[torch.randperm(n)]
        for s in range(0, n, cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            xb, yb = Xt[b], yt[b]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xv)
            vloss = float(loss_fn(vpred, yv).item())

        if vloss + cfg.min_delta < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(Xt).detach().cpu().numpy()


# -----------------------------
# Hyperparameter tuning (Optuna)
# -----------------------------
def hidden_from_params(p: Dict) -> List[int]:
    n_layers = int(p["n_layers"])
    base = int(p["hidden_base"])
    shrink = float(p["shrink"])
    hidden: List[int] = []
    cur = base
    for _ in range(n_layers):
        hidden.append(int(cur))
        cur = max(32, int(cur * shrink))
    return hidden


def tune_arch_and_train(
    X: np.ndarray,
    y: np.ndarray,
    inner_folds: int,
    device: torch.device,
    seed: int,
    trials: int,
    scale_features: bool,
) -> Dict:
    import optuna

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        # architecture
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_base = trial.suggest_int("hidden_base", 128, 1024, step=128)
        shrink = trial.suggest_float("shrink", 0.55, 0.95)
        dropout = trial.suggest_float("dropout", 0.0, 0.35)

        # training
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        max_epochs = trial.suggest_categorical("max_epochs", [80, 120, 200, 300])
        patience = trial.suggest_int("patience", 10, 30)

        hidden = hidden_from_params({"n_layers": n_layers, "hidden_base": hidden_base, "shrink": shrink})
        cfg = TrainCfg(lr=lr, weight_decay=wd, batch_size=bs, max_epochs=max_epochs, patience=patience)

        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            scaler, Xtr_s = maybe_standardize_fit(Xtr, scale_features)
            Xva_s = maybe_standardize_apply(Xva, scaler)

            model = MLPRegressor(X.shape[1], hidden, dropout=dropout)
            model = train_one(model, Xtr_s, ytr, Xva_s, yva, device, cfg)
            pred = predict(model, Xva_s, device)
            fold_scores.append(rmse(yva, pred))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_value"] = float(study.best_value)
    return best


def tune_finetune_train_only(
    X: np.ndarray,
    y: np.ndarray,
    inner_folds: int,
    device: torch.device,
    seed: int,
    trials: int,
    pretrained_state: Dict[str, torch.Tensor],
    hidden: List[int],
    dropout: float,
    freeze_mode: str,
    reset_head: bool,
    scale_features: bool,
) -> Dict:
    import optuna

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        max_epochs = trial.suggest_categorical("max_epochs", [80, 120, 200, 300])
        patience = trial.suggest_int("patience", 10, 40)

        cfg = TrainCfg(lr=lr, weight_decay=wd, batch_size=bs, max_epochs=max_epochs, patience=patience)

        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            scaler, Xtr_s = maybe_standardize_fit(Xtr, scale_features)
            Xva_s = maybe_standardize_apply(Xva, scaler)

            model = MLPRegressor(X.shape[1], hidden, dropout=dropout)
            model.load_state_dict(pretrained_state, strict=True)
            if reset_head:
                model.reset_head()
            set_trainable(model, freeze_mode)

            model = train_one(model, Xtr_s, ytr, Xva_s, yva, device, cfg)
            pred = predict(model, Xva_s, device)
            fold_scores.append(rmse(yva, pred))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_value"] = float(study.best_value)
    return best


# -----------------------------
# Data utils
# -----------------------------
def load_df(path: str, smiles_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"{path}: missing smiles_col='{smiles_col}'")
    if target_col not in df.columns:
        raise ValueError(f"{path}: missing target_col='{target_col}'")
    df = df[[smiles_col, target_col]].dropna()
    df = df.groupby(smiles_col, as_index=False)[target_col].mean()
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_data_path", required=True, type=str)
    ap.add_argument("--low_data_path", default=None, type=str)

    ap.add_argument("--smiles_col", default="SMILES", type=str)
    ap.add_argument("--target_col", default="Cp", type=str)

    ap.add_argument(
        "--fp_method",
        default="maccs",
        type=str,
        choices=["morgan", "rdkit", "maccs", "topologicaltorsion", "atompair", "pe"],
    )
    ap.add_argument("--use_descriptors", action="store_true")
    ap.add_argument("--morgan_radius", default=2, type=int)
    ap.add_argument("--n_bits", default=2048, type=int)
    ap.add_argument("--pe_model_path", default=None, type=str)

    ap.add_argument("--results_root", default="./results", type=str)
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--outer_folds", default=5, type=int)
    ap.add_argument("--inner_folds", default=3, type=int)

    ap.add_argument("--exp_trials", default=50, type=int)
    ap.add_argument("--pretrain_trials", default=80, type=int)
    ap.add_argument("--ft_trials", default=60, type=int)

    ap.add_argument("--do_exp_nested", action="store_true")
    ap.add_argument("--do_pretrain", action="store_true")
    ap.add_argument("--do_ft_nested", action="store_true")

    ap.add_argument("--freeze_mode", default="none", choices=["none", "backbone"])
    ap.add_argument("--reset_head", action="store_true")
    ap.add_argument(
        "--scale_features",
        default=None,
        type=str,
        choices=["true", "false"],
        help="Default: true if use_descriptors else false.",
    )

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    if args.scale_features is None:
        scale_features = bool(args.use_descriptors)
    else:
        scale_features = (args.scale_features.lower() == "true")

    exp_df = load_df(args.exp_data_path, args.smiles_col, args.target_col)
    low_df = load_df(args.low_data_path, args.smiles_col, args.target_col) if args.low_data_path else None

    run_name = (
        f"MLP_{args.fp_method}_{'desc' if args.use_descriptors else 'nod'}"
        f"_{args.target_col}_seed{args.seed}_o{args.outer_folds}_i{args.inner_folds}"
    )
    outdir = os.path.join(args.results_root, run_name)
    ensure_dir(outdir)

    # Featurize EXP
    Xexp = featurize_smiles(
        exp_df[args.smiles_col].tolist(),
        fp_method=args.fp_method,
        use_descriptors=args.use_descriptors,
        morgan_radius=args.morgan_radius,
        n_bits=args.n_bits,
        pe_model_path=args.pe_model_path,
    )
    yexp = exp_df[args.target_col].to_numpy(dtype=np.float32)
    exp_df, Xexp, yexp = filter_valid_rows(exp_df, Xexp, yexp)

    outer = KFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)

    exp_fold_metrics: List[Dict[str, float]] = []
    exp_true_all, exp_pred_all = [], []

    tl_fold_metrics: List[Dict[str, float]] = []
    tl_true_all, tl_pred_all = [], []

    fold_summaries = []

    for fold_id, (tr_idx, te_idx) in enumerate(outer.split(Xexp), start=1):
        fold_dir = os.path.join(outdir, f"outer_fold_{fold_id}")
        ensure_dir(fold_dir)

        X_tr, y_tr = Xexp[tr_idx], yexp[tr_idx]
        X_te, y_te = Xexp[te_idx], yexp[te_idx]

        fold_info = {"fold": fold_id, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}

        # -------------------------
        # 1) EXP-only nested CV
        # -------------------------
        if args.do_exp_nested:
            best_exp = tune_arch_and_train(
                X_tr, y_tr,
                inner_folds=args.inner_folds,
                device=device,
                seed=args.seed + 10 * fold_id,
                trials=args.exp_trials,
                scale_features=scale_features,
            )
            hidden = hidden_from_params(best_exp)
            dropout = float(best_exp["dropout"])
            cfg = TrainCfg(
                lr=float(best_exp["lr"]),
                weight_decay=float(best_exp["weight_decay"]),
                batch_size=int(best_exp["batch_size"]),
                max_epochs=int(best_exp["max_epochs"]),
                patience=int(best_exp["patience"]),
            )

            Xtr2, Xva2, ytr2, yva2 = train_test_split(
                X_tr, y_tr, test_size=0.15, random_state=args.seed + 10 * fold_id
            )
            scaler, Xtr2s = maybe_standardize_fit(Xtr2, scale_features)
            Xva2s = maybe_standardize_apply(Xva2, scaler)
            Xtes = maybe_standardize_apply(X_te, scaler)

            model = MLPRegressor(X_tr.shape[1], hidden, dropout=dropout)
            model = train_one(model, Xtr2s, ytr2, Xva2s, yva2, device, cfg)

            yhat = predict(model, Xtes, device)
            m = metrics(y_te, yhat)

            exp_fold_metrics.append(m)
            for yt, yp in zip(y_te, yhat):
                exp_true_all.append({
                    "fold": fold_id,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                })


            save_json({"best_params": best_exp, "test_metrics": m}, os.path.join(fold_dir, "exp_only.json"))
            save_parity(y_te, yhat, f"EXP-only outer fold {fold_id}", os.path.join(fold_dir, "exp_only_parity.png"))
            fold_info["exp_only"] = {"best_value": best_exp["best_value"], "test": m}

        # -------------------------
        # 2) LOW pretrain + 3) TL fine-tune
        # -------------------------
        if args.do_pretrain and args.do_ft_nested:
            if low_df is None:
                raise SystemExit("TL requested but --low_data_path not provided.")

            exp_smiles_all = set(exp_df.iloc[tr_idx][args.smiles_col]).union(set(exp_df.iloc[te_idx][args.smiles_col]))
            low_work = low_df[~low_df[args.smiles_col].isin(exp_smiles_all)].reset_index(drop=True)

            Xlow = featurize_smiles(
                low_work[args.smiles_col].tolist(),
                fp_method=args.fp_method,
                use_descriptors=args.use_descriptors,
                morgan_radius=args.morgan_radius,
                n_bits=args.n_bits,
                pe_model_path=args.pe_model_path,
            )
            ylow = low_work[args.target_col].to_numpy(dtype=np.float32)
            low_work, Xlow, ylow = filter_valid_rows(low_work, Xlow, ylow)

            if len(low_work) < max(50, args.inner_folds * 10):
                fold_info["tl_skipped"] = f"LOW too small after removing EXP molecules: n_low={len(low_work)}"
                fold_summaries.append(fold_info)
                continue

            best_pre = tune_arch_and_train(
                Xlow, ylow,
                inner_folds=args.inner_folds,
                device=device,
                seed=args.seed + 1000 + fold_id,
                trials=args.pretrain_trials,
                scale_features=scale_features,
            )
            pre_hidden = hidden_from_params(best_pre)
            pre_dropout = float(best_pre["dropout"])
            pre_cfg = TrainCfg(
                lr=float(best_pre["lr"]),
                weight_decay=float(best_pre["weight_decay"]),
                batch_size=int(best_pre["batch_size"]),
                max_epochs=int(best_pre["max_epochs"]),
                patience=int(best_pre["patience"]),
            )

            Xl_tr, Xl_va, yl_tr, yl_va = train_test_split(
                Xlow, ylow, test_size=0.15, random_state=args.seed + 1000 + fold_id
            )
            scaler_low, Xl_tr_s = maybe_standardize_fit(Xl_tr, scale_features)
            Xl_va_s = maybe_standardize_apply(Xl_va, scaler_low)

            pre_model = MLPRegressor(Xlow.shape[1], pre_hidden, dropout=pre_dropout)
            pre_model = train_one(pre_model, Xl_tr_s, yl_tr, Xl_va_s, yl_va, device, pre_cfg)
            pretrained_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}

            save_json(
                {"best_params": best_pre, "arch": {"hidden_dims": pre_hidden, "dropout": pre_dropout}},
                os.path.join(fold_dir, "pretrain_best.json"),
            )

            best_ft = tune_finetune_train_only(
                X_tr, y_tr,
                inner_folds=args.inner_folds,
                device=device,
                seed=args.seed + 2000 + fold_id,
                trials=args.ft_trials,
                pretrained_state=pretrained_state,
                hidden=pre_hidden,
                dropout=pre_dropout,
                freeze_mode=args.freeze_mode,
                reset_head=args.reset_head,
                scale_features=scale_features,
            )
            ft_cfg = TrainCfg(
                lr=float(best_ft["lr"]),
                weight_decay=float(best_ft["weight_decay"]),
                batch_size=int(best_ft["batch_size"]),
                max_epochs=int(best_ft["max_epochs"]),
                patience=int(best_ft["patience"]),
            )

            Xtr2, Xva2, ytr2, yva2 = train_test_split(
                X_tr, y_tr, test_size=0.15, random_state=args.seed + 2000 + fold_id
            )
            scaler_exp, Xtr2s = maybe_standardize_fit(Xtr2, scale_features)
            Xva2s = maybe_standardize_apply(Xva2, scaler_exp)
            Xtes = maybe_standardize_apply(X_te, scaler_exp)

            ft_model = MLPRegressor(X_tr.shape[1], pre_hidden, dropout=pre_dropout)
            ft_model.load_state_dict(pretrained_state, strict=True)
            if args.reset_head:
                ft_model.reset_head()
            set_trainable(ft_model, args.freeze_mode)

            ft_model = train_one(ft_model, Xtr2s, ytr2, Xva2s, yva2, device, ft_cfg)
            yhat = predict(ft_model, Xtes, device)
            m = metrics(y_te, yhat)

            tl_fold_metrics.append(m)
            for yt, yp in zip(y_te, yhat):
                tl_true_all.append({
                    "fold": fold_id,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                })


            save_json(
                {
                    "pretrain_best": best_pre,
                    "ft_best": best_ft,
                    "arch": {"hidden_dims": pre_hidden, "dropout": pre_dropout},
                    "freeze_mode": args.freeze_mode,
                    "reset_head": bool(args.reset_head),
                    "test_metrics": m,
                },
                os.path.join(fold_dir, "transfer_learning.json"),
            )
            save_parity(y_te, yhat, f"TL outer fold {fold_id}", os.path.join(fold_dir, "tl_parity.png"))
            fold_info["transfer_learning"] = {"ft_best_value": best_ft["best_value"], "test": m}

        fold_summaries.append(fold_info)
    # ============================
    # Save final parity datapoints
    # ============================

    if exp_true_all:
        df_exp_parity = pd.DataFrame(exp_true_all)
        df_exp_parity.to_csv(
            os.path.join(outdir, "exp_only_parity_data_all_folds.csv"),
            index=False
        )

    if tl_true_all:
        df_tl_parity = pd.DataFrame(tl_true_all)
        df_tl_parity.to_csv(
            os.path.join(outdir, "tl_parity_data_all_folds.csv"),
            index=False
        )
        
    # Aggregated parity + mean/std reporting
    if exp_fold_metrics:
        y_true = [d["y_true"] for d in exp_true_all]
        y_pred = [d["y_pred"] for d in exp_true_all]
        save_parity(y_true, y_pred, "EXP-only (all outer folds)", os.path.join(outdir, "exp_only_parity_all.png"))
        print_mean_std("EXP-only", exp_fold_metrics)

    if tl_fold_metrics:
        y_true = [d["y_true"] for d in tl_true_all]
        y_pred = [d["y_pred"] for d in tl_true_all]
        save_parity(y_true, y_pred, "Transfer Learning (all outer folds)", os.path.join(outdir, "tl_parity_all.png"))
        print_mean_std("Transfer Learning", tl_fold_metrics)


    exp_mean, exp_std = mean_std_from_fold_metrics(exp_fold_metrics)
    tl_mean, tl_std = mean_std_from_fold_metrics(tl_fold_metrics)

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "n_exp": int(len(exp_df)),
        "fingerprint": args.fp_method,
        "use_descriptors": bool(args.use_descriptors),
        "scale_features": bool(scale_features),
        "fold_summaries": fold_summaries,
        "exp_only": {"fold_metrics": exp_fold_metrics, "mean": exp_mean, "std": exp_std},
        "transfer_learning": {"fold_metrics": tl_fold_metrics, "mean": tl_mean, "std": tl_std},
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    print(f"[DONE] Results saved to: {outdir}")


if __name__ == "__main__":
    main()
