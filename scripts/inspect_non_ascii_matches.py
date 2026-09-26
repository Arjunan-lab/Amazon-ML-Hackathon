import sys, io, re, pandas as pd
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

df_s1 = pd.read_csv("data/train/train_source1.tsv", sep="\t", nrows=20000).set_index("entity_id")
df_s2 = pd.read_csv("data/train/train_source2.tsv", sep="\t", nrows=50000).set_index("entity_id")
df_gt = pd.read_csv("data/train/train_ground_truth.tsv", sep="\t", nrows=20000)

found = 0
for _, row in df_gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in df_s1.index:
        continue
    raw_m = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
    for m in raw_m.split(","):
        m = m.strip()
        if m in df_s2.index:
            s2_row = df_s2.loc[m]
            s2_name = str(s2_row["business_name"])
            if re.search(r'[^\x00-\x7F]', s2_name):
                s1_row = df_s1.loc[s1_id]
                print("--- MATCH FOUND ---")
                print(f"S1: {s1_id} | Name: {s1_row['business_name']} | Address: {s1_row['business_address']}")
                print(f"S2: {m} | Name: {s2_name} | Address: {s2_row['business_address']}")
                found += 1
                if found >= 5:
                    break
    if found >= 5:
        break
