import re
import nltk
import logging
import json
import uuid
from difflib import get_close_matches, Differ
from langdetect import detect
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Private Use Area markers to fence content we must not translate
START_KEEP = "\uE000"
END_KEEP = "\uE001"

TIME_PATTERNS = [
    (r'\b(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM))\b', "__TIME_COLON_%d__"),
    (r'\b(\d{1,2}\s?(?:am|pm|AM|PM))\b', "__TIME_%d__"),
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', "__MONTH_%d__"),
]

def mock_db_update():
    """Mocks database update (stateless)."""
    logger.info("Database update mocked. No history tracked.")
    pass

def mock_rag_lookup(text: str) -> str:
    """Mocks RAG lookup for contextual information."""
    logger.info("RAG lookup mocked. Returning no context.")
    return ""

def load_combined_kb(files: List[str]) -> Dict[str, str]:
    """Load and combine multiple KB files."""
    combined_kb = {}
    def normalize_key(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())
    for kb_path in files:
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        en = item.get("English") or item.get("english") or item.get("source")
                        zh = item.get("Simplified_Chinese") or item.get("simplified_chinese") or item.get("Traditional_Chinese") or item.get("traditional_chinese") or item.get("target")
                        if en and zh:
                            key = normalize_key(en)
                            if key not in combined_kb:
                                combined_kb[key] = zh
        except Exception as e:
            logger.warning(f"KB load failed for {kb_path}: {e}")
    return combined_kb

def kb_lookup_fuzzy(kb: Dict[str, str], key: str, cutoff=0.85) -> str | None:
    if not key:
        return None
    nk = re.sub(r"\s+", " ", key.strip().lower())
    if nk in kb:
        return kb[nk]
    matches = get_close_matches(nk, kb.keys(), n=1, cutoff=cutoff)
    if matches:
        return kb[matches[0]]
    return None

def setup_nltk() -> bool:
    """Ensure NLTK punkt is installed; fallback to regex if failed."""
    try:
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.data.find('tokenizers/punkt')
            return True
        except Exception as e:
            logger.warning(f"NLTK punkt download failed: {e}")
            return False

def clean_text(text: str) -> str:
    """Clean text while preserving punctuation, CJK characters, and line structure."""
    if not text:
        return text
    # Replace non-breaking spaces with regular spaces
    text = re.sub(r'\u00A0', ' ', text)
    # Remove control characters (but preserve newlines)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    # Preserve newlines: split by newlines, clean each line, then rejoin
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Normalize spaces/tabs within the line (collapse multiple spaces/tabs to single space)
        cleaned_line = re.sub(r'[ \t]+', ' ', line)
        # Remove trailing spaces but preserve the line structure
        cleaned_line = cleaned_line.rstrip()
        cleaned_lines.append(cleaned_line)
    # Rejoin lines, preserving newline structure
    result = '\n'.join(cleaned_lines)
    # Only strip leading/trailing whitespace from the entire text, not individual lines
    return result.strip()

# def preserve_formatting_elements(text: str) -> Tuple[str, Dict[str, str]]:
#     """Preserve URLs, emails, and citations (including bracketed citations like [0001])."""
#     preserved_elements = {}
#     tmp_text = text
    
#     # Preserve URLs first (before other patterns)
#     url_pattern = r'https?://[^\s]+'
#     urls = re.findall(url_pattern, tmp_text)
#     for i, url in enumerate(urls):
#         placeholder = f"__URL_{i}__"
#         tmp_text = tmp_text.replace(url, placeholder)
#         preserved_elements[placeholder] = url
    
#     # Preserve emails
#     email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     emails = re.findall(email_pattern, tmp_text)
#     for i, email in enumerate(emails):
#         placeholder = f"__EMAIL_{i}__"
#         tmp_text = tmp_text.replace(email, placeholder)
#         preserved_elements[placeholder] = email
    
#     # Enhanced citation preservation - process in reverse order to avoid index shifting
#     # Pattern 1: Full citations like "U.S. Non-Provisional Application No. 15/947,182, filed April 6, 2018"
#     full_citation_patterns = [
#         # Full U.S. citations with filed date
#         r'(?:U\.S\.|United States)\s+(?:Non-?Provisional|Provisional)?\s*(?:Application|Patent)\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?,\s*filed\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
#         # U.S. patent/applications without filed date
#         r'(?:U\.S\.|United States)\s+(?:Non-?Provisional|Provisional)?\s*(?:Application|Patent)\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?',
#         # Patent numbers
#         r'U\.S\.\s+Pat\.?\s+No\.?\s*\d{1,3}(?:,\d{3})*(?:,\d+)?',
#         # Application numbers
#         r'(?:Application|App)\.?\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?',
#         # Filed dates standalone
#         r'filed\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
#         # Claims references
#         r'claims?\s+the\s+benefit\s+of\s+(?:U\.S\.|United States)\s+[^.]+\d+[/\-]\d+[^.]*',
#     ]
    
#     citation_idx = 0
#     all_matches = []
#     for pattern in full_citation_patterns:
#         for match in re.finditer(pattern, tmp_text, re.IGNORECASE):
#             all_matches.append((match.start(), match.end(), match.group(0)))
    
#     # Sort by position (descending) to replace from end to start
#     all_matches.sort(key=lambda x: x[0], reverse=True)
    
#     for start, end, citation_text in all_matches:
#         placeholder = f"{START_KEEP}CITATION_{citation_idx}{END_KEEP}"
#         tmp_text = tmp_text[:start] + placeholder + tmp_text[end:]
#         preserved_elements[placeholder] = citation_text
#         citation_idx += 1
    
#     # Pattern 2: Simple bracketed citations like [0001], [ 0001 ], [0001 ], [ 0001]
#     # Allow optional spaces inside the brackets; process in reverse order
#     bracket_matches = list(re.finditer(r'\[\s*\d+\s*\]', tmp_text))
#     for match in sorted(bracket_matches, key=lambda m: m.start(), reverse=True):
#         citation_text = match.group(0)
#         placeholder = f"{START_KEEP}CITATION_{citation_idx}{END_KEEP}"
#         tmp_text = tmp_text[:match.start()] + placeholder + tmp_text[match.end():]
#         preserved_elements[placeholder] = citation_text
#         citation_idx += 1
    
#     return tmp_text, preserved_elements


START_KEEP = "<<KEEP>>"
END_KEEP = "<</KEEP>>"


def _make_cit_placeholder(idx: int) -> str:
    """
    Create a unique placeholder for each preserved element.
    Uses readable ASCII markers instead of Unicode control characters.
    """
    unique = uuid.uuid4().hex[:8]
    return f"{START_KEEP}CIT_{idx}_{unique}{END_KEEP}"



def preserve_formatting_elements(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Preserve URLs, emails, full citation phrases, and bracketed citations.
    Returns (protected_text, preserved_elements_map).
    """
    preserved_elements = {}
    tmp_text = text

    # 1) URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, tmp_text)
    for i, url in enumerate(urls):
        placeholder = f"{START_KEEP}URL_{i}_{uuid.uuid4().hex[:6]}{END_KEEP}"
        tmp_text = tmp_text.replace(url, placeholder)
        preserved_elements[placeholder] = url

    # 2) Emails
    email_pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    emails = re.findall(email_pattern, tmp_text)
    for i, email in enumerate(emails):
        placeholder = f"{START_KEEP}EMAIL_{i}_{uuid.uuid4().hex[:6]}{END_KEEP}"
        tmp_text = tmp_text.replace(email, placeholder)
        preserved_elements[placeholder] = email

    # 3) Full citation patterns (as before) - collect and replace from end->start
    full_citation_patterns = [
        r'(?:U\.S\.|United States)\s+(?:Non-?Provisional|Provisional)?\s*(?:Application|Patent)\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?,\s*filed\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'(?:U\.S\.|United States)\s+(?:Non-?Provisional|Provisional)?\s*(?:Application|Patent)\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?',
        r'U\.S\.\s+Pat\.?\s+No\.?\s*\d{1,3}(?:,\d{3})*(?:,\d+)?',
        r'(?:Application|App)\.?\s+No\.?\s*\d+[/\-]\d+(?:,\d+)?',
        r'filed\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'claims?\s+the\s+benefit\s+of\s+(?:U\.S\.|United States)\s+[^.]+\d+[/\-]\d+[^.]*',
    ]
    all_matches = []
    for pat in full_citation_patterns:
        for m in re.finditer(pat, tmp_text, flags=re.IGNORECASE):
            all_matches.append((m.start(), m.end(), m.group(0)))
    all_matches.sort(key=lambda x: x[0], reverse=True)
    cit_idx = 0
    for start, end, cit_text in all_matches:
        ph = _make_cit_placeholder(cit_idx)
        tmp_text = tmp_text[:start] + ph + tmp_text[end:]
        preserved_elements[ph] = cit_text
        cit_idx += 1

    # 4) Bracketed citations — robust: include ASCII, fullwidth and a few common unicode variants
    # This pattern captures [0001], [Ref-01], 【0001】, 〔0001〕, and similar
    bracket_chars_open = r'[\[\uFF3B\u3010\u3014\u3016\uFF5B\uFF08\uFF5F\u2772]'
    bracket_chars_close = r'[\]\uFF3D\u3011\u3015\u3017\uFF5D\uFF09\uFF60\u2773]'
    bracket_pattern = rf'{bracket_chars_open}\s*[A-Za-z0-9\-\._\s,]+\s*{bracket_chars_close}'
    # Find all bracket matches and replace from end to start
    bracket_matches = list(re.finditer(bracket_pattern, tmp_text))
    for m in sorted(bracket_matches, key=lambda mm: mm.start(), reverse=True):
        ph = _make_cit_placeholder(cit_idx)
        preserved_elements[ph] = m.group(0)
        tmp_text = tmp_text[:m.start()] + ph + tmp_text[m.end():]
        cit_idx += 1

    # 5) Simple numeric citations on their own or label+number patterns (line starts or inline)
    # e.g., "Claim 1", "Fig. 3", "FIG 4", localized words may occur but we catch general patterns
    # We'll match common English labels and numeric-only sequences too (only short matches)
    label_pattern = r'\b(?:Claim|Claims|Fig|Figure|Fig\.|Example|Ex|Ref|Reference|Claim:|Claim/|Claim/)\s*\d{1,4}\b'
    for m in list(re.finditer(label_pattern, tmp_text, flags=re.IGNORECASE)):
        ph = _make_cit_placeholder(cit_idx)
        preserved_elements[ph] = m.group(0)
        tmp_text = tmp_text[:m.start()] + ph + tmp_text[m.end():]
        cit_idx += 1

    # Also protect numeric-only tokens that look like standalone citation lines (e.g., lines that are only "[1]" or "1.")
    # We'll scan line-by-line for short tokens that are just numbers possibly with surrounding punctuation
    new_lines = []
    for line in tmp_text.splitlines(keepends=True):
        stripped = line.strip()
        if re.match(r'^(?:' + bracket_chars_open + r')?\s*\d{1,4}\s*(?:' + bracket_chars_close + r')?[.:]?\s*$', stripped):
            # replace the numeric token inside the line
            ph = _make_cit_placeholder(cit_idx)
            preserved_elements[ph] = stripped
            # keep trailing newline if present
            suffix = '\n' if line.endswith('\n') else ''
            new_lines.append(ph + suffix)
            cit_idx += 1
        else:
            new_lines.append(line)
    tmp_text = ''.join(new_lines)

    return tmp_text, preserved_elements


def preserve_time_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    preserved = {}
    tmp = text
    idx = 0
    for pat, placeholder_fmt in TIME_PATTERNS:
        for m in re.finditer(pat, tmp):
            ph = placeholder_fmt % idx
            preserved[ph] = m.group(1)
            tmp = tmp.replace(m.group(1), ph, 1)
            idx += 1
    return tmp, preserved

# def restore_formatting_elements(text: str, preserved_elements: Dict[str, str]) -> str:
#     """Restore preserved formatting elements, handling cases where placeholders may have been translated."""
#     # First, try to restore exact placeholder matches
#     sorted_placeholders = sorted(preserved_elements.items(), key=lambda x: len(x[0]), reverse=True)
    
#     for placeholder, original in sorted_placeholders:
#         if placeholder in text:
#             text = text.replace(placeholder, original)
    
#     # Then handle translated citations (placeholders that were translated by the model)
#     # Extract all citation placeholders in order
#     citation_placeholders = [(k, v) for k, v in sorted(preserved_elements.items()) if "CITATION_" in k]
    
#     if citation_placeholders:
#         # Handle mangled citation placeholders (e.g., <unk>CITATION_1<H_0__ or CITATION_1 with other text)
#         # Pattern 1: Mangled citations like <unk>CITATION_N, CITATION_N<..., etc.
#         for placeholder, original in citation_placeholders:
#             # Extract citation number from placeholder (e.g., CITATION_1 -> 1)
#             citation_num_match = re.search(r'CITATION_(\d+)', placeholder)
#             if citation_num_match:
#                 citation_num = citation_num_match.group(1)
                
#                 # Check if original is a bracketed citation like [0001]
#                 is_bracketed_citation = original.strip().startswith('[') and original.strip().endswith(']')
                
#                 # Pattern to match mangled citations: <unk>CITATION_N, CITATION_N<..., CITATION_N with any trailing text
#                 # More aggressive patterns to catch various mangled forms
#                 mangled_patterns = [
#                     rf'<unk>\s*CITATION_{citation_num}<[^>]*>',  # <unk>CITATION_1<H_0__>
#                     rf'<unk>\s*CITATION_{citation_num}[^A-Za-z0-9]',  # <unk>CITATION_1< or <unk>CITATION_1 
#                     rf'CITATION_{citation_num}<[^>]*>',  # CITATION_1<H_0__>
#                     rf'CITATION_{citation_num}[^A-Za-z0-9]',  # CITATION_1 followed by non-alphanumeric
#                     rf'CITATION_{citation_num}\b',  # CITATION_1 standalone (last resort, word boundary)
#                 ]
#                 for pattern in mangled_patterns:
#                     if re.search(pattern, text):
#                         # Replace the entire mangled pattern with the original citation
#                         text = re.sub(pattern, original, text)
#                         break
                
#                 # Special handling for bracketed citations [0001] - they should remain unchanged
#                 # Try to restore even if brackets got translated or altered
#                 if is_bracketed_citation:
#                     # Extract the number from the original citation (e.g., [0001] -> 0001)
#                     citation_num_in_brackets = re.search(r'\[(\d+)\]', original)
#                     if citation_num_in_brackets:
#                         num = citation_num_in_brackets.group(1)
#                         # Try to find mangled versions of bracketed citations and restore them
#                         # Pattern: [num], [ num ], translated brackets, etc.
#                         # Also handle cases where the number got translated (e.g., [0] should restore to [0001])
#                         bracketed_mangled_patterns = [
#                             (rf'\[\s*{num}\s*\]', True),  # [0001] or [ 0001 ] with spaces - exact match
#                             (rf'\[{num}\]', True),  # [0001] without spaces - exact match
#                             (rf'\(\s*{num}\s*\)', True),  # (0001) - brackets got translated
#                             (f'\\{{\\s*{num}\\s*\\}}', True),  # {0001} - curly brackets
#                             (rf'\[\s*\d\s*\]', False),  # [0] - single digit (likely a mangled citation)
#                         ]
#                         for pattern, is_exact in bracketed_mangled_patterns:
#                             matches = list(re.finditer(pattern, text))
#                             if matches:
#                                 # For exact matches, replace all occurrences
#                                 # For single-digit fallback, only replace first occurrence to avoid false positives
#                                 matches_to_replace = matches if is_exact else [matches[0]] if matches else []
#                                 # Process from end to start to avoid index shifting
#                                 for match in reversed(matches_to_replace):
#                                     # Replace with original format preserving leading zeros
#                                     text = text[:match.start()] + original.strip() + text[match.end():]
#                                 if matches_to_replace:
#                                     break
        
#         # Pattern 2: Translated citation words for all supported languages
#         # These patterns catch cases where the model translates citation placeholders
#         citation_patterns = [
#             r'Zitat\s*[-–—]\s*\d+',  # German "Zitat - 0"
#             r'Citation\s*[-–—]\s*\d+',  # English/French "Citation - 0"
#             r'Citación\s*[-–—]\s*\d+',  # Spanish "Citación - 0"
#             r'Citazione\s*[-–—]\s*\d+',  # Italian "Citazione - 0"
#             r'Citat\s*[-–—]\s*\d+',  # Swedish/Danish "Citat - 0"
#             r'Lainaus\s*[-–—]\s*\d+',  # Finnish "Lainaus - 0"
#             r'Citaat\s*[-–—]\s*\d+',  # Dutch "Citaat - 0"
#             r'引用\s*[-–—]\s*\d+',  # Chinese "引用 - 0" (citation)
#             r'引用番号\s*[-–—]\s*\d+',  # Japanese "引用番号 - 0" (citation number)
#             r'인용\s*[-–—]\s*\d+',  # Korean "인용 - 0" (citation)
#         ]
        
#         # Find all translated citation patterns in the text
#         translated_citations = []
#         for pattern in citation_patterns:
#             for match in re.finditer(pattern, text, re.IGNORECASE):
#                 translated_citations.append((match.start(), match.end(), match.group(0)))
        
#         # Sort by position
#         translated_citations.sort(key=lambda x: x[0])
        
#         # Replace translated citations with original citations in order
#         # Process from end to start to avoid index shifting
#         for i, (start, end, _) in enumerate(reversed(translated_citations)):
#             if i < len(citation_placeholders):
#                 _, original = citation_placeholders[-(i+1)]  # Get in reverse order
#                 text = text[:start] + original + text[end:]
    
#     # Final generic fallback: normalize any translated citation labels like
#     # "Zitat - 0", "Citation 12", "Citación-3", etc., into bracket form [12]
#     # This catches cases where no placeholders were preserved at all.
#     # Supports all languages: German, English, French, Spanish, Italian, Swedish, Danish, Finnish, Dutch, Chinese, Japanese, Korean
#     text = re.sub(r'(?i)\b(zitat|citation|citaat|citación|citazione|lainaus|引用|引用番号|인용)\s*[-–—]?\s*(\d+)\b',
#                   lambda m: f"[{m.group(2)}]", text)

#     # Remove any keep markers that leaked through
#     text = text.replace(START_KEEP, "").replace(END_KEEP, "")

#     return text


def restore_formatting_elements(text: str, preserved_elements: Dict[str, str]) -> str:
    """
    Restore preserved placeholders. Handles mangled or split KEEP markers and placeholders.
    Normalizes localized citation formats and fixes partially broken markers.
    """
    # 1) Direct replacements (longest placeholders first)
    for placeholder, original in sorted(preserved_elements.items(), key=lambda x: len(x[0]), reverse=True):
        if placeholder in text:
            text = text.replace(placeholder, original)

    # 2) Handle mangled placeholder patterns
    for placeholder, original in preserved_elements.items():
        core_match = re.search(r'CIT_(\d+)_([0-9a-fA-F]{1,8})', placeholder)
        if not core_match:
            continue
        num, uuidfrag = core_match.groups()
        patterns = [
            rf'{re.escape("CIT_" + num + "_" + uuidfrag)}',
            rf'CIT[_\-\s]*{num}[_\-\s]*{uuidfrag}',
            rf'<unk>\s*CIT[_\-\s]*{num}',
            rf'CIT[_\-\s]*{num}\b',
            rf'\bCIT{num}\b',
        ]
        replaced_any = False
        for p in patterns:
            if re.search(p, text):
                text = re.sub(p, original, text)
                replaced_any = True
                break
        if not replaced_any:
            if re.search(r'CIT[_\-\s]*' + re.escape(num), text):
                text = re.sub(r'CIT[_\-\s]*' + re.escape(num), original, text)

    # 3) Normalize localized "Citation - N" formats
    localized_label_words = [
        'zitat', 'citation', 'citación', 'citazione', 'citat', 'lainaus', 'citaat',
        '引用', '引用番号', '인용', 'kysymys', 'referencia', 'referenz', 'referência', 'figura'
    ]
    localized_regex = rf'(?i)\b(?:' + '|'.join(map(re.escape, localized_label_words)) + r')\s*[:\-–—]?\s*(\d{{1,4}})\b'
    text = re.sub(localized_regex, lambda m: f"[{m.group(1)}]", text)

    # 4) Convert loose sequences like " / 0" → "[0]"
    text = re.sub(r'(?<=\b)(?:/|\-)\s*(\d{1,4})\b', lambda m: f"[{m.group(1)}]", text)

    # 5) Fix mangled KEEP markers (e.g., << KEEP>>, <</KEEP>, << KEEP >, <</ KEEP>>)
    text = re.sub(r'<+\s*KEEP\s*>+>?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<+/?\s*KEEP\s*>+>?', '', text, flags=re.IGNORECASE)

    # 6) Also catch partially duplicated endings like <</KEEP>EEP>>
    text = re.sub(r'EEP>+>?', '', text)

    return text


def restore_time_tokens(text: str, preserved: Dict[str, str]) -> str:
    for ph, orig in preserved.items():
        text = text.replace(ph, orig)
    return text

def intelligent_chunk_text(text: str, max_len: int = 400) -> List[str]:
    if len(text) <= max_len:
        return [text]
    sentences = safe_sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) > max_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def safe_sent_tokenize(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    if text.count("\n\n") > 3:
        parts = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) < 200:
                parts.append(para)
            else:
                parts.extend(_tokenize_long_para(para))
        return [p.strip() for p in parts if p.strip()]
    return _tokenize_long_para(text)

def _tokenize_long_para(para: str) -> List[str]:
    from app import setup_nltk  # Avoid circular import
    nltk_available = setup_nltk()
    if nltk_available:
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(para)
        except Exception:
            return re.split(r'(?<=[\.\?\!])\s+', para)
    return re.split(r'(?<=[\.\?\!])\s+', para)

def is_translation_valid(translated_text: str, original_text: str) -> bool:
    if not translated_text or not translated_text.strip():
        logger.debug("Translation invalid: empty result")
        return False
    if len(translated_text.strip()) < 2:
        logger.debug("Translation invalid: too short")
        return False
    words = translated_text.split()
    if len(words) > 5:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        if words:
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.8:
                logger.debug(f"Translation invalid: excessive repetition ({max_count}/{len(words)})")
                return False
    error_patterns = [
        r'(.)\1{20,}',
        r'(\w+\s+)\1{10,}',
    ]
    for pattern in error_patterns:
        if re.search(pattern, translated_text):
            logger.debug(f"Translation invalid: error pattern detected")
            return False
    logger.debug(f"Translation valid: '{translated_text[:50]}...'")
    return True

def highlight_deltas(source: str, target: str) -> List[Dict[str, str]]:
    source_tokens = source.split()
    target_tokens = target.split()
    differ = Differ()
    deltas = []
    for diff in differ.compare(source_tokens, target_tokens):
        status_char = diff[0]
        token = diff[2:].strip()
        if not token:
            continue
        if status_char == ' ':
            deltas.append({'status': 'kept', 'text': token})
        elif status_char == '-':
            deltas.append({'status': 'removed', 'text': token})
        elif status_char == '+':
            deltas.append({'status': 'added', 'text': token})
    return deltas

def clean_json_response(json_str: str) -> str:
    """
    Clean JSON string from Llama responses by removing:
    - Single-line comments (// ...)
    - Multi-line comments (/* ... */)
    - Trailing commas before closing braces/brackets
    """
    # Remove single-line comments (// ...)
    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
    
    # Remove multi-line comments (/* ... */)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Remove trailing commas before closing braces or brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    return json_str

def get_language_code(lang_name: str) -> str:
    from .config import LANG_OPTIONS
    normalized_lang = lang_name.lower().strip()
    return LANG_OPTIONS.get(normalized_lang, lang_name)

def preserve_and_restore_capitalization(original_text: str, translated_text: str) -> str:
    """
    Preserve and restore capitalization from original text to translated text.
    Specifically handles ALL-CAPS headings and sentences.
    
    Strategy:
    1. Detect if original text is ALL-CAPS (likely heading)
    2. Preserve ALL-CAPS for headings in translation
    3. Maintain sentence case for regular text
    4. Handle mixed case (partial translations)
    """
    if not original_text or not translated_text:
        return translated_text
    
    result = translated_text
    
    # 1. Check if entire original text is ALL-CAPS (heading)
    orig_stripped = original_text.strip()
    if orig_stripped and len(orig_stripped) > 2:
        # Check if original is mostly/all uppercase (heading), ignoring non-letters and whitespace
        letters_only = ''.join(ch for ch in orig_stripped if ch.isalpha())
        uppercase_chars = sum(1 for c in letters_only if c.isupper())
        total_alpha = len(letters_only)
        if total_alpha > 0:
            uppercase_ratio = uppercase_chars / total_alpha
            # If >80% uppercase, treat as heading and preserve ALL-CAPS
            if uppercase_ratio > 0.8:
                result = result.upper()
                return result
    
    # 2. Check if original starts with ALL-CAPS words (likely heading)
    # Match patterns like "RELATED APPLICATIONS", "TECHNICAL FIELD"
    first_words = orig_stripped.split()[:3]  # First 3 words
    if first_words:
        def norm_letters(w: str) -> str:
            return ''.join(ch for ch in w if ch.isalpha())
        all_caps_start = all(len(norm_letters(word)) > 1 and norm_letters(word).isupper() for word in first_words if norm_letters(word))
        if all_caps_start:
            # This looks like a heading - preserve ALL-CAPS
            result = result.upper()
            return result
    
    # 3. Handle sentence-level capitalization
    # Capitalize first letter after sentence endings
    def capitalize_first_letter(match):
        return match.group(0)[:-1] + match.group(1).upper()
    
    result = re.sub(r'(?:^|\.\s+)([a-z])', capitalize_first_letter, result)
    result = re.sub(r'(?:^|!\s+)([a-z])', capitalize_first_letter, result)
    result = re.sub(r'(?:^|\?\s+)([a-z])', capitalize_first_letter, result)
    
    # 4. Handle mixed case where original had ALL-CAPS but translation is mixed
    # Split into sentences and check each
    orig_sentences = re.split(r'([.!?]\s+)', original_text)
    trans_sentences = re.split(r'([.!?]\s+)', result)
    
    for i, orig_sent in enumerate(orig_sentences):
        if i < len(trans_sentences):
            orig_sent_stripped = orig_sent.strip()
            # If original sentence was ALL-CAPS (likely heading)
            if orig_sent_stripped and len(orig_sent_stripped) > 3:
                uppercase_chars = sum(1 for c in orig_sent_stripped if c.isupper())
                total_alpha = sum(1 for c in orig_sent_stripped if c.isalpha())
                if total_alpha > 0:
                    uppercase_ratio = uppercase_chars / total_alpha
                    # If >80% uppercase, preserve ALL-CAPS in translation
                    if uppercase_ratio > 0.8:
                        trans_sentences[i] = trans_sentences[i].upper()
    
    result = ''.join(trans_sentences)
    
    # 5. Ensure proper sentence capitalization
    result = re.sub(r'(?:^|\.\s+)([a-z])', capitalize_first_letter, result)
    
    return result