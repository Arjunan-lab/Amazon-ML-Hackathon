"""Industry-Standard Text and Address Normalization Engine (Stage 1).

Implements:
1. Zero-dependency, offline unified Brahmic transliteration across all 9 Indian scripts
   (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Odia, Gurmukhi).
2. Unicode NFKD decomposition & diacritic stripping for French and European records.
3. Word-bounded legal and corporate entity suffix stripping across US, India, and France
   with strict empty-string guardrails to prevent over-trimming.
4. Comprehensive multi-country address component standardization (US, India, France).
5. Numerical token extraction and address invariant anchoring.
"""

from typing import Dict, List, Optional, Set, Union
import re
import unicodedata
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Indic Script Transliteration Tables (9 Major Languages)
# -----------------------------------------------------------------------------
INDIC_OFFSET_MAP: Dict[int, str] = {
    # Independent Vowels
    0x02: "n",   # Anusvara
    0x03: "h",   # Visarga
    0x05: "a",   0x06: "aa",  0x07: "i",   0x08: "ee",
    0x09: "u",   0x0A: "oo",  0x0B: "ri",  0x0E: "e",
    0x0F: "e",   0x10: "ai",  0x12: "o",   0x13: "o",   0x14: "au",

    # Consonants
    0x15: "k",   0x16: "kh",  0x17: "g",   0x18: "gh",  0x19: "ng",
    0x1A: "ch",  0x1B: "chh", 0x1C: "j",   0x1D: "jh",  0x1E: "ny",
    0x1F: "t",   0x20: "th",  0x21: "d",   0x22: "dh",  0x23: "n",
    0x24: "t",   0x25: "th",  0x26: "d",   0x27: "dh",  0x28: "n",   0x29: "nn",
    0x2A: "p",   0x2B: "ph",  0x2C: "b",   0x2D: "bh",  0x2E: "m",
    0x2F: "y",   0x30: "r",   0x31: "rr",  0x32: "l",   0x33: "ll",  0x34: "lll",
    0x35: "v",   0x36: "sh",  0x37: "sh",  0x38: "s",   0x39: "h",

    # Dependent Vowel Signs (Matras)
    0x3E: "a",   0x3F: "i",   0x40: "ee",  0x41: "u",   0x42: "oo",
    0x43: "ri",  0x46: "e",   0x47: "e",   0x48: "ai",  0x4A: "o",
    0x4B: "o",   0x4C: "au",
    0x49: "o",   # Candra O (e.g. Hindi 'मॉ')
    0x4D: "",    # Virama / Halant (suppresses implicit vowel)
}

INDIC_BASES: List[int] = [
    0x0900,  # Devanagari (Hindi, Marathi, Sanskrit)
    0x0980,  # Bengali / Assamese
    0x0A00,  # Gurmukhi (Punjabi)
    0x0A80,  # Gujarati
    0x0B00,  # Odia
    0x0B80,  # Tamil
    0x0C00,  # Telugu
    0x0C80,  # Kannada
    0x0D00,  # Malayalam
]

# Explicit Regional State/City Name Mappings (when appearing in native scripts)
REGIONAL_LOCATION_MAP: Dict[str, str] = {
    "தமிழ்நாடு": "tamil nadu",
    "சென்னை": "chennai",
    "கர்நாಟಕ": "karnataka",
    "ಕರ್ನಾಟಕ": "karnataka",
    "ಬೆಂಗಳೂರು": "bangalore",
    "ತెలಂಗಾಣ": "telangana",
    "తెలంగాణ": "telangana",
    "హైదరాబాద్": "hyderabad",
    "കേരളം": "kerala",
    "കൊച്ചി": "kochi",
    "পশ্চিমবঙ্গ": "west bengal",
    "কলকাতা": "kolkata",
    "ગુજરાત": "gujarat",
    "અમદાવાદ": "ahmedabad",
    "महाराष्ट्र": "maharashtra",
    "मुंबई": "mumbai",
    "दिल्ली": "delhi",
    "ਪੰਜਾਬ": "punjab",
    "ଓଡ଼ିଶା": "odisha",
}


# -----------------------------------------------------------------------------
# 2. Corporate & Legal Suffix Specifications
# -----------------------------------------------------------------------------
LEGAL_SUFFIXES: List[str] = [
    # US / UK / International
    r"\bincorporated\b", r"\binc\b",
    r"\bcorporation\b", r"\bcorp\b",
    r"\blimited liability company\b", r"\bllc\b",
    r"\blimited liability partnership\b", r"\bllp\b",
    r"\bcompany\b", r"\bco\b",
    r"\blimited\b", r"\bltd\b",
    r"\bplc\b", r"\bpllc\b",
    
    # India (English & Common Transliterations)
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt\b", r"\bprivate\b",
    r"\benterprises\b", r"\benterprise\b",
    r"\bassociates\b", r"\band sons\b", r"\band co\b",
    r"\bindustries\b", r"\btraders\b", r"\btechnologies\b", r"\bsolutions\b",
    r"\bpraivet limited\b", r"\bpiraivet limitet\b", r"\bpraivet\b", r"\blimitet\b",
    r"\bknshltnnchi\b", r"\bknsltents\b",

    # France (Test Set)
    r"\bsociete anonyme\b", r"\bsa\b",
    r"\bsociete a responsabilite limitee\b", r"\bsarl\b",
    r"\bsociete par actions simplifiee\b", r"\bsas\b",
    r"\bsociete par actions simplifiee unipersonnelle\b", r"\bsasu\b",
    r"\bsociete civile immobiliere\b", r"\bsci\b",
    r"\bentreprise unipersonnelle a responsabilite limitee\b", r"\beurl\b",
    r"\bgroupement d interet economique\b", r"\bgie\b",
    r"\bsociete\b", r"\bste\b",
]

_SUFFIX_REGEX: re.Pattern = re.compile(r"|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)


# -----------------------------------------------------------------------------
# 3. Comprehensive Address Standardization Dictionary
# -----------------------------------------------------------------------------
ADDRESS_MAP: Dict[str, str] = {
    # US / UK / International
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bcir\b": "circle",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bno\b": "number",
    r"#": " number ",

    # France
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bblv\b": "boulevard",
    r"\ball\b": "allee",
    r"\bche\b": "chemin",
    r"\bchem\b": "chemin",
    r"\bimp\b": "impasse",
    r"\bcrs\b": "cours",
    r"\brte\b": "route",
    r"\bbis\b": "bis",

    # India (Municipalities, Landmarks & Areas)
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bnfr\b": "near",
    r"\bextn\b": "extension",
    r"\bcol\b": "colony",
    r"\btq\b": "taluk",
    r"\btaluka\b": "taluk",
    r"\bdist\b": "district",
    r"\bsec\b": "sector",
    r"\bsect\b": "sector",
    r"\bph\b": "phase",
    r"\bstg\b": "stage",
    r"\bpos\b": "post office",
    r"\bpo\b": "post office",
    r"\bkh no\b": "khasra number",
    r"\bplot no\b": "plot number",
    r"\bh no\b": "house number",
}

_ADDRESS_COMPILED_MAP = [(re.compile(p, flags=re.IGNORECASE), repl) for p, repl in ADDRESS_MAP.items()]


# -----------------------------------------------------------------------------
# 4. Core Transformation Functions
# -----------------------------------------------------------------------------
def transliterate_indic(text: str) -> str:
    """Converts Indic script characters across 9 languages into phonetic Roman text."""
    if not text:
        return ""
    
    # Check for direct regional state/city matches first
    for reg_key, reg_val in REGIONAL_LOCATION_MAP.items():
        if reg_key in text:
            text = text.replace(reg_key, f" {reg_val} ")

    res = []
    for ch in text:
        cp = ord(ch)
        matched = False
        for base in INDIC_BASES:
            if base <= cp < base + 0x80:
                offset = cp - base
                if offset in INDIC_OFFSET_MAP:
                    res.append(INDIC_OFFSET_MAP[offset])
                matched = True
                break
        if not matched:
            res.append(ch)
    return "".join(res)


def strip_accents(text: str) -> str:
    """Normalizes unicode characters and strips combining accents (e.g., é -> e, ç -> c)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def clean_text(text: str) -> str:
    """Universal cleaning pipeline: transliterates Indic text, strips accents, cleans symbols."""
    if not text or not isinstance(text, str):
        return ""
    # 1. Transliterate Indic characters to Roman phonetics
    text = transliterate_indic(text)
    # 2. Decompose Unicode and strip French/European accents
    text = strip_accents(text).lower()
    # 3. Contextual symbol replacements
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("%", " percent ")
    text = text.replace("+", " plus ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    # 4. Strip quotes and brackets
    text = re.sub(r"[«»“”\"'<>\[\](){}]", " ", text)
    # 5. Keep only alphanumeric characters and single spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Normalizes business name by removing corporate legal suffixes with guardrails.

    Safety Constraint:
        If suffix removal empties the string or leaves fewer than 2 characters
        (e.g., entity is literally named 'The Company'), fallback to the cleaned text
        to ensure no record ever becomes empty.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    
    # Remove word-bounded legal suffixes
    stripped = _SUFFIX_REGEX.sub(" ", cleaned)
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # Guardrail: Revert if stripping destroyed the entire name
    if len(stripped) < 2:
        return cleaned

    return stripped


def normalize_address(address: str) -> str:
    """Normalizes address by standardizing street/landmark abbreviations and whitespace."""
    cleaned = clean_text(address)
    if not cleaned:
        return ""

    for regex_pat, repl in _ADDRESS_COMPILED_MAP:
        cleaned = regex_pat.sub(repl, cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_numbers(text: str) -> Set[str]:
    """Extracts all numerical tokens (pincodes, suite numbers, street numbers, door numbers).
    
    Captures numeric portions from alphanumeric codes (e.g., '448' from '448A').
    """
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


# -----------------------------------------------------------------------------
# 5. Vectorized Batch Normalizer for Pandas DataFrames
# -----------------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Stage 1 normalization across all business entity columns in a DataFrame.

    Adds:
        - clean_name: Suffix-stripped, transliterated, accent-free business name.
        - clean_addr: Standardized canonical address string.
        - raw_clean_name: Accent-free business name preserving suffixes.
        - num_tokens: Extracted set of numbers (pincodes, door numbers).
    """
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)

    df["clean_name"] = df["business_name"].apply(normalize_business_name)
    df["clean_addr"] = df["business_address"].apply(normalize_address)
    df["raw_clean_name"] = df["business_name"].apply(clean_text)
    
    # Extract numerical tokens from combined name + address
    df["num_tokens"] = (df["business_name"] + " " + df["business_address"]).apply(extract_numbers)
    return df
