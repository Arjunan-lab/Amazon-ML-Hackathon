import sys, io, os
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath("."))
from code.business_entity_resolution.src.normalization import normalize_business_name, normalize_address

print("Hindi:          ", normalize_business_name("मॉडर्न फाइनेंस"))
print("Tamil:          ", normalize_business_name("ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட்"))
print("Kannada:        ", normalize_business_name("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"))
print("French Name:    ", normalize_business_name("Fédération de Velo SAS"))
print("French Address: ", normalize_address("175 Boulevard du Président Franklin Roosevelt"))
