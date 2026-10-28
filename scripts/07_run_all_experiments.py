import sys
import os
import argparse
import subprocess
import logging

sys.path.insert(0, os.path.abspath("src"))
from cross_source_trust.utils import setup_logger, load_yaml

logger = setup_logger("07_run_all_experiments")

def run_command(cmd_args: list[str], desc: str):
    logger.info(f"==> Starting step: {desc} ({' '.join(cmd_args)})")
    res = subprocess.run(cmd_args, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Step failed: {desc}")
        logger.error(f"STDOUT:\n{res.stdout}")
        logger.error(f"STDERR:\n{res.stderr}")
        raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")
    logger.info(f"Step completed: {desc}")
    if res.stdout.strip():
        for line in res.stdout.strip().splitlines()[-10:]:
            logger.info(f"   [out] {line}")

def main():
    parser = argparse.ArgumentParser(description="Master script to run entire research pipeline end-to-end.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--all-seeds", action="store_true", help="Run all 10 seeds (0-9)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seeds = cfg["experiment"]["seeds"] if args.all_seeds else args.seeds
    seed_str = [str(s) for s in seeds]

    logger.info("=" * 70)
    logger.info("LAUNCHING COMPLETE EXPERIMENT REPRODUCTION PIPELINE")
    logger.info(f"Target seeds: {seed_str}")
    logger.info("=" * 70)

    python_exe = sys.executable

    # Step 1: Download & Verify
    run_command([python_exe, "scripts/01_download_data.py"], "01 Data Download")

    # Step 2: Validate Raw Data
    run_command([python_exe, "scripts/02_validate_data.py"], "02 Data Validation")

    # Step 3: Preprocess & Chronological Partitioning
    run_command([python_exe, "scripts/03_preprocess_data.py"], "03 Data Preprocessing")

    # Step 4: Model Training
    run_command([python_exe, "scripts/04_train_models.py", "--seeds"] + seed_str, "04 Model Training")

    # Step 5: Detection Evaluation
    run_command([python_exe, "scripts/05_run_detection.py", "--seeds"] + seed_str, "05 Detection Evaluation")

    # Step 6: Dispatch Simulation
    run_command([python_exe, "scripts/06_run_dispatch.py", "--seeds"] + seed_str, "06 Battery Dispatch Simulation")

    # Step 7: Manuscript Outputs Generation (Figures & Tables)
    run_command([python_exe, "scripts/08_generate_manuscript_outputs.py"], "07 Generate Figures & Tables")

    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS EXECUTED AND OUTPUTS GENERATED SUCCESSFULLY!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
