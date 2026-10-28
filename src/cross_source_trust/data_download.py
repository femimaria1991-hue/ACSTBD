import os
import sys
import time
import struct
import subprocess
import urllib.request
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("cross_source_trust.data_download")

ZENODO_PSML_URL = "https://zenodo.org/record/5130612/files/PSML.zip?download=1"
PJM_ZONE_1_OFFSET = 2001284337
PJM_ZONE_1_COMP_SIZE = 58866913
PJM_ZONE_1_FN = "PSML/Minute-level Load and Renewable/PJM_zone_1_.csv"
PSML_CD_OFFSET = 5178768338
PSML_CD_SIZE = 390861

def download_psml_zone_data(raw_dir: str = "data/raw", force: bool = False) -> str:
    """
    Downloads and extracts PJM_zone_1_.csv from the official PSML Zenodo repository
    using HTTP range requests for efficient, targeted download (58.8 MB vs 5.18 GB).
    """
    os.makedirs(raw_dir, exist_ok=True)
    target_csv = os.path.join(raw_dir, "PJM_zone_1_.csv")
    if os.path.exists(target_csv) and not force:
        logger.info(f"PSML file already exists: {target_csv} ({os.path.getsize(target_csv)} bytes)")
        return target_csv

    zip_path = os.path.join(raw_dir, "pjm_zone_1.zip")
    part_path = zip_path + ".part"
    total_entry_len = 30 + len(PJM_ZONE_1_FN) + PJM_ZONE_1_COMP_SIZE

    logger.info(f"Downloading {PJM_ZONE_1_FN} from Zenodo ({total_entry_len / 1024 / 1024:.2f} MB)...")
    
    start_byte = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    chunk_size = 4 * 1024 * 1024

    with open(part_path, "ab") as f:
        while start_byte < total_entry_len:
            end_byte = min(start_byte + chunk_size, total_entry_len) - 1
            req = urllib.request.Request(
                ZENODO_PSML_URL,
                headers={"Range": f"bytes={PJM_ZONE_1_OFFSET + start_byte}-{PJM_ZONE_1_OFFSET + end_byte}"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                f.write(data)
            start_byte += len(data)
            logger.info(f"PSML download progress: {start_byte / 1024 / 1024:.2f} / {total_entry_len / 1024 / 1024:.2f} MB")

    # Fetch central directory entry to construct valid standalone zip
    req = urllib.request.Request(
        ZENODO_PSML_URL,
        headers={"Range": f"bytes={PSML_CD_OFFSET}-{PSML_CD_OFFSET + PSML_CD_SIZE - 1}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        cd_bytes = resp.read()

    idx = 0
    cd_entry = None
    while idx < len(cd_bytes):
        if cd_bytes[idx:idx+4] != b"PK\x01\x02":
            break
        flen, elen, clen = struct.unpack("<HHH", cd_bytes[idx+28:idx+34])
        fn = cd_bytes[idx+46:idx+46+flen].decode("utf-8", errors="ignore")
        if fn == PJM_ZONE_1_FN:
            cd_entry = bytearray(cd_bytes[idx:idx+46+flen+elen+clen])
            cd_entry[42:46] = struct.pack("<I", 0) # set local header offset to 0
            break
        idx += 46 + flen + elen + clen

    if cd_entry is None:
        raise RuntimeError(f"Could not find CD entry for {PJM_ZONE_1_FN} in Zenodo central directory")

    with open(part_path, "ab") as f:
        f.write(cd_entry)
        eocd = struct.pack("<IHHHHIIH", 0x06054b50, 0, 0, 1, 1, len(cd_entry), total_entry_len, 0)
        f.write(eocd)

    if os.path.exists(zip_path):
        os.remove(zip_path)
    os.rename(part_path, zip_path)

    # Extract with 7-Zip or system tool
    seven_zip = "C:\\Program Files\\7-Zip\\7z.exe"
    if os.path.exists(seven_zip):
        cmd = [seven_zip, "x", zip_path, f"-o{os.path.abspath(raw_dir)}", "-y"]
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        # Fallback to tar
        subprocess.run(["tar", "-xf", zip_path, "-C", raw_dir], check=True)

    extracted_path = os.path.join(raw_dir, PJM_ZONE_1_FN.replace("/", os.sep))
    if os.path.exists(extracted_path):
        if os.path.exists(target_csv):
            os.remove(target_csv)
        os.rename(extracted_path, target_csv)
    
    logger.info(f"Successfully extracted {target_csv} ({os.path.getsize(target_csv)} bytes)")
    return target_csv

def download_pjm_pricing_data(
    raw_dir: str = "data/raw",
    api_key: str = None,
    start_date: str = "2018-01-01",
    end_date: str = "2020-12-31"
) -> tuple[str, bool]:
    """
    Downloads official PJM Day-Ahead Hourly LMP data.
    
    If PJM_API_KEY is supplied or in os.environ, queries the official feed via gridstatus.
    If official data file data/raw/pjm_da_hourly_lmps_2018_2020.csv already exists, loads it.
    Otherwise, documents official instructions and creates an interim empirical proxy
    (clearly flagged as unverified proxy) so downstream steps remain operational.
    
    Returns: (file_path, is_verified_real_data)
    """
    os.makedirs(raw_dir, exist_ok=True)
    target_csv = os.path.join(raw_dir, "pjm_da_hourly_lmps_2018_2020.csv")

    if os.path.exists(target_csv):
        logger.info(f"PJM Day-Ahead price file found: {target_csv}")
        return target_csv, True

    key = api_key or os.environ.get("PJM_API_KEY")
    if key:
        logger.info("PJM API Key detected. Fetching Day-Ahead Hourly LMP via gridstatus...")
        try:
            import gridstatus
            pjm = gridstatus.PJM(api_key=key)
            df = pjm.get_lmp(
                start=start_date,
                end=end_date,
                market="DAY_AHEAD_HOURLY"
            )
            df.to_csv(target_csv, index=False)
            logger.info(f"Downloaded verified PJM LMP data to {target_csv}")
            return target_csv, True
        except Exception as e:
            logger.error(f"Failed to fetch data using PJM API: {e}")

    # Official instructions for user
    logger.warning("=" * 70)
    logger.warning("OFFICIAL PJM DAY-AHEAD MARKET DATA ACCESS INSTRUCTIONS:")
    logger.warning("1. Register for a free PJM API account at: https://accountmanager.pjm.com/")
    logger.warning("2. Generate a primary subscription key in the PJM API Portal.")
    logger.warning("3. Set environment variable: set PJM_API_KEY=<your_key>")
    logger.warning("   OR manually download Day-Ahead Hourly LMPs (2018-01-01 to 2020-12-31)")
    logger.warning("   from https://dataminer2.pjm.com/feed/da_hrl_lmps and save to:")
    logger.warning(f"   {os.path.abspath(target_csv)}")
    logger.warning("4. To allow full pipeline execution in the interim, a clearly marked")
    logger.warning("   empirical proxy file will be generated (FLAGGED: UNVERIFIED PROXY).")
    logger.warning("=" * 70)

    proxy_csv = os.path.join(raw_dir, "pjm_da_hourly_lmps_proxy_unverified.csv")
    if not os.path.exists(proxy_csv):
        # Diurnal PJM empirical price distribution based on published PJM market averages (~$25 - $45/MWh)
        dr = pd.date_range("2018-01-01 00:00:00", "2020-12-31 23:00:00", freq="1h")
        rng = np.random.default_rng(42)
        hour_pattern = np.array([
            24, 22, 21, 20, 21, 25, 32, 38, 42, 40,
            38, 36, 35, 34, 35, 37, 43, 49, 48, 44,
            40, 36, 31, 26
        ], dtype=float)
        base_price = np.tile(hour_pattern, len(dr) // 24 + 1)[:len(dr)]
        noise = rng.normal(0, 4.0, size=len(dr))
        # Add seasonal heatwave / winter peak spikes
        season = 5.0 * np.sin(2 * np.pi * dr.dayofyear / 365.25)
        price = np.clip(base_price + season + noise, 10.0, 150.0)
        
        df_proxy = pd.DataFrame({
            "datetime_beginning_ept": dr,
            "total_lmp_da": price,
            "is_unverified_proxy": True
        })
        df_proxy.to_csv(proxy_csv, index=False)
        logger.info(f"Generated interim proxy file: {proxy_csv}")

    return proxy_csv, False

def run_download_pipeline(force: bool = False):
    """Executes the complete data acquisition process."""
    psml_path = download_psml_zone_data(force=force)
    price_path, is_verified = download_pjm_pricing_data()
    return {
        "psml_path": psml_path,
        "price_path": price_path,
        "price_is_verified": is_verified
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_download_pipeline()
    print("Download results:", res)
