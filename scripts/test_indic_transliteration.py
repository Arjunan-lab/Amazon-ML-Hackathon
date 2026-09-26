"""Test unified Indic script transliteration using Unicode block offsets."""

import sys, io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Base offset mapping for Brahmic Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam)
INDIC_OFFSET_MAP = {
    # Vowels
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    
    # Consonants
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "l", 0x33: "ll", 0x34: "lll",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    
    # Signs & Matras
    0x02: "n", 0x03: "h",
    0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo",
    0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "", # Virama (halant / suppress vowel)
    0x49: "o", # Candra O (used in Hindi 'मॉडर्न')
}

INDIC_BASES = [
    0x0900, # Devanagari
    0x0980, # Bengali
    0x0A00, # Gurmukhi
    0x0A80, # Gujarati
    0x0B00, # Odia
    0x0B80, # Tamil
    0x0C00, # Telugu
    0x0C80, # Kannada
    0x0D00, # Malayalam
]

def transliterate_indic(text: str) -> str:
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
                else:
                    matched = True
                    break
        if not matched:
            res.append(ch)
    return "".join(res)

test_cases = [
    ("Hindi", "मॉडर्न फाइनेंस"),
    ("Hindi", "लाइफ कंसल्टेंट्स प्राइवेट लिमिटेड"),
    ("Tamil", "ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட்"),
    ("Tamil", "தமிழ்நாடு"),
    ("Kannada", "ಸನ್ ವೆಂಚರ್ಸ್ ಟೆಕ್ನಾಲಜೀಸ್"),
    ("Kannada", "ಕರ್ನಾಟಕ"),
    ("Telugu", "తెలంగాణ"),
    ("Gujarati", "ગુજરાત"),
    ("Bengali", "পশ্চিমবঙ্গ"),
]

for lang, text in test_cases:
    print(f"[{lang}] {text} -> '{transliterate_indic(text)}'")
