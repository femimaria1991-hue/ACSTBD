import sys
import os
import argparse

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.preprocessing import preprocess_and_synchronize
from cross_source_trust.utils import setup_logger, load_yaml

def main():
    parser = argparse.ArgumentParser(description="Synchronize and preprocess raw datasets.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

    logger = setup_logger("03_preprocess_data")
    cfg = load_yaml(args.config)["dataset"]

    logger.info("Executing preprocessing and synchronization pipeline...")
    res = preprocess_and_synchronize(
        psml_csv="data/raw/PJM_zone_1_.csv",
        price_csv="data/raw/pjm_da_hourly_lmps_2018_2020.csv",
        interim_dir="data/interim",
        processed_dir="data/processed",
        load_peak_mw=cfg["scaling"]["load_peak_mw"],
        renewable_peak_mw=cfg["scaling"]["renewable_peak_mw"]
    )
    logger.info("Preprocessing completed successfully.")
    logger.info(f"Train subset: {res['train_path']}")
    logger.info(f"Calibration subset: {res['cal_path']}")
    logger.info(f"Test subset: {res['test_path']}")

if __name__ == "__main__":
    main()
