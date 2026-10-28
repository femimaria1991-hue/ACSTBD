import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.abspath("src"))

from cross_source_trust.data_download import run_download_pipeline
from cross_source_trust.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Download PSML and PJM datasets.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist.")
    parser.add_argument("--api-key", type=str, default=None, help="PJM Data Miner API subscription key.")
    args = parser.parse_args()

    logger = setup_logger("01_download_data")
    logger.info("Executing dataset acquisition pipeline...")
    
    if args.api_key:
        os.environ["PJM_API_KEY"] = args.api_key

    results = run_download_pipeline(force=args.force)
    logger.info("Data download completed.")
    logger.info(f"PSML raw path: {results['psml_path']}")
    logger.info(f"PJM price path: {results['price_path']} (Verified: {results['price_is_verified']})")

if __name__ == "__main__":
    main()
