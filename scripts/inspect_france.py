import sys, io, pandas as pd
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_csv("data/test/test_source1.tsv", sep="\t")
france_df = df[df["country"] == "France"].head(5)
for idx, row in france_df.iterrows():
    print(f"ID: {row['entity_id']} | Name: {row['business_name']} | Address: {row['business_address']}")
