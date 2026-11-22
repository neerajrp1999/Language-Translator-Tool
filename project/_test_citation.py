import sys
from backend.utils import clean_text, preserve_formatting_elements, restore_formatting_elements

original = """RELATED APPLICATIONS\n\n[0001] This application claims the benefit of U.S. Non-Provisional Application No. 15/947,182, filed April 6, 2018."""
proc, preserved = preserve_formatting_elements(clean_text(original))

# Simulated Dutch model output with broken citation forms
translated = """GERELATEERDE TOEPASSINGEN\n\nCITATEN 0Deze aanvraag beweert het voordeel van U.S. Non-Provisional Application No. 15/947,182<unk>H_0__ 6, 2018."""
restored = restore_formatting_elements(translated, preserved)
print(restored)
