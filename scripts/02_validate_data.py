import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.data_validation import validate_raw_psml
from cross_source_trust.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Validate downloaded raw datasets.")
    parser.add_argument("--psml-csv", type=str, default="data/raw/PJM_zone_1_.csv")
    args = parser.parse_args()

    logger = setup_logger("02_validate_data")
    logger.info(f"Validating raw PSML file: {args.psml_csv}")

    if not os.path.exists(args.psml_csv):
        logger.error(f"File not found: {args.psml_csv}. Please run scripts/01_download_data.py first.")
        sys.exit(1)

    df = pd.read_csv(args.psml_csv, usecols=['time', 'load_power', 'wind_power', 'solar_power', 'Temperature'])
    res = validate_raw_psml(df)
    logger.info(f"Raw data schema validation: {res['status']}")
    logger.info(f"Available channels: {res['columns']}")

if __name__ == "__main__":
    main()
