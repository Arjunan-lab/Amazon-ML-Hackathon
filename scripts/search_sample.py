import sys, io, pandas as pd
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

for src_name, path in [("Test S2", "data/test/test_source2.tsv"), ("Test S3", "data/test/test_source3.tsv")]:
    print(f"\nSearching in {src_name}...")
    for chunk in pd.read_csv(path, sep="\t", chunksize=500000):
        matches = chunk[chunk["business_name"].str.contains("ZNB Club|Team Ecole", case=False, na=False)]
        for _, r in matches.iterrows():
            print(f"[{src_name}] ID: {r['entity_id']} | Name: {r['business_name']} | Address: {r['business_address']} | Country: {r['country']}")
