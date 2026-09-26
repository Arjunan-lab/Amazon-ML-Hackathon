"""Demo Stage 1 Normalization Engine on real competition test data."""

import sys, io, os
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from code.business_entity_resolution.src.normalization import preprocess_dataframe

for country, path in [
    ("France", "data/test/test_source1.tsv"),
    ("India", "data/test/test_source2.tsv"),
    ("US", "data/test/test_source3.tsv"),
]:
    df = pd.read_csv(path, sep="\t", nrows=10000)
    subset = df[df["country"] == country].head(3)
    clean_subset = preprocess_dataframe(subset)

    print("=" * 70)
    print(f"REAL DATA NORMALIZATION DEMO: [{country}] from {os.path.basename(path)}")
    print("=" * 70)
    for idx, r in clean_subset.iterrows():
        print(f"Original Name:    {r['business_name']}")
        print(f"Cleaned Name:     {r['clean_name']}")
        print(f"Original Address: {r['business_address']}")
        print(f"Cleaned Address:  {r['clean_addr']}")
        print(f"Numeric Tokens:   {r['num_tokens']}")
        print("-" * 70)
