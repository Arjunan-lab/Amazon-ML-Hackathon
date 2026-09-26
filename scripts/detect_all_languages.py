"""Comprehensive Unicode script and language detector for test_source2.tsv and test_source3.tsv."""

import sys
import io
import unicodedata
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def get_char_script(char: str) -> str:
    """Classifies a unicode character into its script/language category."""
    cp = ord(char)
    if cp < 128:
        return "Latin (ASCII/English)"
    elif 0x00A0 <= cp <= 0x024F:
        return "Latin Extended (French / European Accents)"
    elif 0x0900 <= cp <= 0x097F:
        return "Devanagari (Hindi / Marathi)"
    elif 0x0980 <= cp <= 0x09FF:
        return "Bengali / Assamese"
    elif 0x0A00 <= cp <= 0x0A7F:
        return "Gurmukhi (Punjabi)"
    elif 0x0A80 <= cp <= 0x0AFF:
        return "Gujarati"
    elif 0x0B00 <= cp <= 0x0B7F:
        return "Odia"
    elif 0x0B80 <= cp <= 0x0BFF:
        return "Tamil"
    elif 0x0C00 <= cp <= 0x0C7F:
        return "Telugu"
    elif 0x0C80 <= cp <= 0x0CFF:
        return "Kannada"
    elif 0x0D00 <= cp <= 0x0D7F:
        return "Malayalam"
    elif 0x0600 <= cp <= 0x06FF:
        return "Arabic / Urdu"
    elif 0x0400 <= cp <= 0x04FF:
        return "Cyrillic (Russian / Slavic)"
    elif 0x4E00 <= cp <= 0x9FFF:
        return "CJK (Chinese)"
    elif 0x3040 <= cp <= 0x30FF:
        return "Japanese (Kana)"
    elif 0xAC00 <= cp <= 0xD7AF:
        return "Korean (Hangul)"
    elif 0x0370 <= cp <= 0x03FF:
        return "Greek"
    elif 0x2000 <= cp <= 0x206F or 0x20A0 <= cp <= 0x20CF:
        return "Punctuation / Currency Symbols"
    else:
        try:
            return unicodedata.name(char).split()[0]
        except Exception:
            return "Other"


def analyze_file(file_path: str, max_rows: int = 1000000):
    print("=" * 70)
    print(f"ANALYZING: {file_path}")
    print(f"Scanning up to {max_rows:,} rows across business_name and business_address...")
    print("=" * 70)

    script_counts = Counter()
    examples_by_script = defaultdict(list)
    country_script_counts = defaultdict(Counter)

    chunksize = 200000
    rows_processed = 0

    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize, nrows=max_rows, dtype=str):
        for _, row in chunk.iterrows():
            rows_processed += 1
            name = str(row.get("business_name", "") or "")
            addr = str(row.get("business_address", "") or "")
            country = str(row.get("country", "") or "Unknown")
            full_text = f"{name} {addr}"

            # Identify which non-ASCII scripts are present in this record
            record_scripts = set()
            for ch in full_text:
                script = get_char_script(ch)
                if script not in ("Latin (ASCII/English)", "Punctuation / Currency Symbols", "Other"):
                    record_scripts.add(script)

            for s in record_scripts:
                script_counts[s] += 1
                country_script_counts[country][s] += 1
                if len(examples_by_script[s]) < 3:
                    examples_by_script[s].append({
                        "id": row.get("entity_id", ""),
                        "name": name,
                        "address": addr,
                        "country": country
                    })

    print(f"\nCompleted scan of {rows_processed:,} records.")
    print("-" * 70)
    print(f"{'Script / Language Family':<45} | {'Occurrences':<12} | {'% of Records'}")
    print("-" * 70)
    for script, count in script_counts.most_common():
        pct = (count / rows_processed) * 100
        print(f"{script:<45} | {count:<12,} | {pct:.2f}%")

    print("\nBreakdown by Country:")
    for country, counts in country_script_counts.items():
        print(f"\n  [{country}]:")
        for s, c in counts.most_common(5):
            print(f"    - {s}: {c:,} records")

    print("\nReal Sample Records for each Script / Language:")
    for script, ex_list in examples_by_script.items():
        print(f"\n  >>> {script}:")
        for ex in ex_list:
            print(f"      [{ex['country']}] ID: {ex['id']} | Name: {ex['name']} | Addr: {ex['address']}")


if __name__ == "__main__":
    analyze_file("data/test/test_source2.tsv", max_rows=500000)
    analyze_file("data/test/test_source3.tsv", max_rows=500000)
