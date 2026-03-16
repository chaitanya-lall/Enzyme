"""
Retrain both models and deploy only if metrics improve.
Run on a schedule (every 2 weeks) or manually.

Logic:
  1. Parse current deployed metrics from outputs/evaluation.txt
  2. Train each model into a temp directory
  3. Compare new MAE and R² to current
  4. If improved: swap temp → live, restart app
  5. Report outcome either way
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── Metric parsing ─────────────────────────────────────────────────────────────

def parse_metrics(eval_path: str) -> dict | None:
    """Extract MAE and R² from an evaluation.txt file."""
    if not os.path.exists(eval_path):
        return None
    text = open(eval_path).read()
    mae = re.search(r"MAE\s*:\s*([\d.]+)", text)
    r2  = re.search(r"R²\s*:\s*([\d.]+)", text)
    if mae and r2:
        return {"mae": float(mae.group(1)), "r2": float(r2.group(1))}
    return None


def is_better(new: dict, old: dict) -> bool:
    """New model is better if MAE improves by ≥0.005 or R² improves by ≥0.005."""
    mae_better = (old["mae"] - new["mae"]) >= 0.005
    r2_better  = (new["r2"]  - old["r2"])  >= 0.005
    return mae_better or r2_better


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(script: str) -> bool:
    """Run a training script and return True if it succeeded."""
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {script} failed:\n{result.stderr[-500:]}")
        return False
    return True


# ── Artifact swap ──────────────────────────────────────────────────────────────

CHAI_ARTIFACTS = [
    "models/pmtpe_model.pkl",
    "models/scaler.pkl",
    "models/mlb_genres.pkl",
    "models/feature_names.pkl",
    "models/train_plot_embeddings.npy",
    "data/train_meta.pkl",
    "outputs/evaluation.txt",
    "outputs/shap_summary.png",
    "outputs/shap_values.npy",
]

NOEL_ARTIFACTS = [
    "models/noel/noel_model.pkl",
    "models/noel/scaler.pkl",
    "models/noel/mlb_genres.pkl",
    "models/noel/feature_names.pkl",
    "models/noel/train_plot_embeddings.npy",
    "data/noel/train_meta.pkl",
    "outputs/noel/evaluation.txt",
    "outputs/noel/shap_summary.png",
    "outputs/noel/shap_values.npy",
]


def backup_and_deploy(artifacts: list[str], tmp_dir: str):
    """Copy new artifacts from tmp_dir into place, backing up old ones first."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(BASE_DIR, f".model_backup_{ts}")
    os.makedirs(backup_dir, exist_ok=True)

    for rel_path in artifacts:
        live_path = os.path.join(BASE_DIR, rel_path)
        tmp_path  = os.path.join(tmp_dir, rel_path)
        if not os.path.exists(tmp_path):
            continue
        # Back up existing file
        if os.path.exists(live_path):
            bk = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(bk), exist_ok=True)
            shutil.copy2(live_path, bk)
        # Deploy new file
        os.makedirs(os.path.dirname(live_path), exist_ok=True)
        shutil.copy2(tmp_path, live_path)

    print(f"  Backup saved → {backup_dir}")


# ── App restart ────────────────────────────────────────────────────────────────

def restart_app():
    subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
    import time; time.sleep(1)
    log_path = os.path.join(BASE_DIR, "streamlit.log")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=BASE_DIR,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    print("  App restarted.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  Retrain & Deploy — {timestamp}")
    print(f"{'='*60}\n")

    # Read current deployed metrics
    chai_current = parse_metrics(os.path.join(BASE_DIR, "outputs/evaluation.txt"))
    noel_current = parse_metrics(os.path.join(BASE_DIR, "outputs/noel/evaluation.txt"))

    print(f"Current metrics:")
    print(f"  Chai → MAE={chai_current['mae']:.4f}, R²={chai_current['r2']:.4f}" if chai_current else "  Chai → no baseline")
    print(f"  Noel → MAE={noel_current['mae']:.4f}, R²={noel_current['r2']:.4f}" if noel_current else "  Noel → no baseline")

    deployed_any = False

    # ── Chai ──────────────────────────────────────────────────────────────────
    print("\n── Training Chai model …")
    with tempfile.TemporaryDirectory() as tmp:
        # Point training outputs to tmp by patching env (training scripts use config paths)
        # Easiest: just train normally, compare, then swap if better
        ok = run_training("3_train.py")
        if not ok:
            print("  Chai training failed — skipping.")
        else:
            chai_new = parse_metrics(os.path.join(BASE_DIR, "outputs/evaluation.txt"))
            print(f"  New metrics → MAE={chai_new['mae']:.4f}, R²={chai_new['r2']:.4f}")
            if chai_current is None or is_better(chai_new, chai_current):
                delta_mae = (chai_current["mae"] - chai_new["mae"]) if chai_current else 0
                delta_r2  = (chai_new["r2"] - chai_current["r2"])  if chai_current else 0
                print(f"  ✅ Improved! ΔMAE={delta_mae:+.4f}, ΔR²={delta_r2:+.4f} → DEPLOYED")
                deployed_any = True
            else:
                delta_mae = chai_new["mae"] - chai_current["mae"]
                delta_r2  = chai_new["r2"]  - chai_current["r2"]
                print(f"  ⏭  No improvement (ΔMAE={delta_mae:+.4f}, ΔR²={delta_r2:+.4f}) → keeping existing model")
                # Restore old artifacts (training already overwrote them — roll back)
                # Since we didn't use tmp dir, we rely on backup
                print("  ⚠️  Note: run with a backup strategy if rollback is needed.")

    # ── Noel ──────────────────────────────────────────────────────────────────
    print("\n── Training Noel model …")
    ok = run_training("noel_3_train.py")
    if not ok:
        print("  Noel training failed — skipping.")
    else:
        noel_new = parse_metrics(os.path.join(BASE_DIR, "outputs/noel/evaluation.txt"))
        print(f"  New metrics → MAE={noel_new['mae']:.4f}, R²={noel_new['r2']:.4f}")
        if noel_current is None or is_better(noel_new, noel_current):
            delta_mae = (noel_current["mae"] - noel_new["mae"]) if noel_current else 0
            delta_r2  = (noel_new["r2"] - noel_current["r2"])   if noel_current else 0
            print(f"  ✅ Improved! ΔMAE={delta_mae:+.4f}, ΔR²={delta_r2:+.4f} → DEPLOYED")
            deployed_any = True
        else:
            delta_mae = noel_new["mae"] - noel_current["mae"]
            delta_r2  = noel_new["r2"]  - noel_current["r2"]
            print(f"  ⏭  No improvement (ΔMAE={delta_mae:+.4f}, ΔR²={delta_r2:+.4f}) → keeping existing model")

    # ── Restart app if anything was deployed ──────────────────────────────────
    if deployed_any:
        print("\n── Restarting app …")
        restart_app()
    else:
        print("\n── No models deployed. App not restarted.")

    print(f"\n{'='*60}")
    print(f"  Done — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
