"""Create a template for real accident-segment coordinates.

The accident dataset does not include GPS coordinates. This script deliberately
never invents latitude/longitude values. Use coordinates from a verified route
GIS file or an authoritative map/source.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PATH = PROCESSED_DIR / "accident_data_prepared.csv"
TEMPLATE_PATH = PROCESSED_DIR / "location_segments_template.csv"
COLUMNS = [
    "Accident_Location_A", "chainage_start_km", "chainage_end_km",
    "latitude", "longitude", "segment_name",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Prepared data file not found: {DATA_PATH}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(columns=COLUMNS).to_csv(TEMPLATE_PATH, index=False)
print(f"Created a blank template: {TEMPLATE_PATH}")
print("Fill it with one row per real road segment, then save as location_segments.csv.")
print("Required fields: location category code, start/end chainage (km), real latitude/longitude, segment name.")
print("Important: Accident_Location_A is a category such as urban/rural, not a route ID.")
print("If routes overlap in chainage, the source data needs a highway/route identifier to map uniquely.")