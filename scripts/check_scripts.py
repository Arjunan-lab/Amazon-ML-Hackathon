import sys, io, re, pandas as pd
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def non_ascii_ratio(path, col="business_name", sample_size=100000):
    df = pd.read_csv(path, sep="\t", nrows=sample_size, usecols=[col, "country"])
    non_ascii = df[col].astype(str).apply(lambda s: bool(re.search(r'[^\x00-\x7F]', s)))
    devanagari = df[col].astype(str).apply(lambda s: bool(re.search(r'[\u0900-\u097F]', s)))
    
    print(f"[{path}] Sample {sample_size:,} rows:")
    print(f"  Non-ASCII names: {non_ascii.sum():,} ({non_ascii.mean()*100:.2f}%)")
    print(f"  Devanagari names: {devanagari.sum():,} ({devanagari.mean()*100:.2f}%)")
    by_c = df.groupby("country")[col].apply(lambda s: s.apply(lambda x: bool(re.search(r'[\u0900-\u097F]', str(x)))).mean()*100)
    print("  Devanagari % by country:\n", by_c)

non_ascii_ratio("data/train/train_source1.tsv")
non_ascii_ratio("data/train/train_source2.tsv")
