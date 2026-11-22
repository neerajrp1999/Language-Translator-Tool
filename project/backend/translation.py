import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import docx
import fitz  # PyMuPDF for PDF
import re
from pptx import Presentation
from docx.shared import Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from copy import deepcopy
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from pdf2docx import Converter
import subprocess
import shutil
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from .config import MAX_RETRIES, USE_XML_METHOD, LIBREOFFICE_PATH, LANG_OPTIONS
from .utils import clean_text, preserve_formatting_elements, preserve_time_tokens, restore_formatting_elements, restore_time_tokens, is_translation_valid, intelligent_chunk_text, kb_lookup_fuzzy, get_language_code, preserve_and_restore_capitalization
from .file_handlers import read_docx_enhanced, read_pdf_enhanced, read_pptx_enhanced
from .state import AppState
from langdetect import detect
import pymupdf  # New import for PyMuPDF (fitz)
from docx.shared import Pt
logger = logging.getLogger(__name__)

def load_translation_model(app_state: AppState):
    MODEL_NAME = "facebook/m2m100_418M"
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not check_device_memory(device):
            logger.warning("Insufficient GPU memory, falling back to CPU")
            device = "cpu"
        model.to(device)
        logger.info(f"Model loaded on {device}")
        app_state.model = model
        app_state.tokenizer = tokenizer
        app_state.device = device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def clear_device_cache(device: str):
    """Clear cache for the specified device (CPU or GPU)."""
    if device == "cuda":
        torch.cuda.empty_cache()

def get_device_memory_info(device: str) -> Tuple[int, int]:
    """Get memory usage for the specified device."""
    if device == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        return allocated_memory, total_memory
    return 0, 0

def check_device_memory(device: str) -> bool:
    """Check if sufficient memory is available for the specified device."""
    if device == "cuda":
        allocated, total = get_device_memory_info(device)
        free_memory = total - allocated
        return free_memory > 1 * 1024 * 1024 * 1024  # Require at least 1GB free
    return True

def translate_text_enhanced(text: str, target_lang: str, src_lang: str = 'en', max_retries: int = 2, app_state: AppState = None) -> str:
    if not text or not text.strip():
        return text
    try:
        cleaned_text = clean_text(text)
        processed_text, preserved_elements = preserve_formatting_elements(cleaned_text)
        processed_text, preserved_time = preserve_time_tokens(processed_text)
        for attempt in range(max_retries):
            try:
                app_state.tokenizer.src_lang = src_lang
                encoded = app_state.tokenizer(
                    processed_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(app_state.device)
                with torch.no_grad():
                    target_lang_code = get_language_code(target_lang)
                    forced_bos_token_id = app_state.tokenizer.lang_code_to_id.get(target_lang_code, None)
                    if forced_bos_token_id is None:
                        generated_tokens = app_state.model.generate(
                            **encoded,
                            max_length=1024,
                            num_beams=3,
                            length_penalty=0.8,
                            early_stopping=True,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.2
                        )
                    else:
                        generated_tokens = app_state.model.generate(
                            **encoded,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=1024,
                            num_beams=3,
                            length_penalty=0.8,
                            early_stopping=True,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.2
                        )
                translated = app_state.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                if is_translation_valid(translated, processed_text):
                    final_translation = restore_formatting_elements(translated, preserved_elements)
                    final_translation = restore_time_tokens(final_translation, preserved_time)
                    logger.debug(f"Translation successful on attempt {attempt + 1}")
                    return final_translation
                else:
                    logger.warning(f"Poor quality translation detected, retrying... (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        logger.info("Using best available translation despite quality issues")
                        final_translation = restore_formatting_elements(translated, preserved_elements)
                        final_translation = restore_time_tokens(final_translation, preserved_time)
                        return final_translation
            except Exception as e:
                logger.error(f"Translation error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.warning(f"All translation attempts failed, returning original text")
                    restored_text = restore_formatting_elements(processed_text, preserved_elements)
                    restored_text = restore_time_tokens(restored_text, preserved_time)
                    return restored_text
        return text
    except Exception as e:
        logger.error(f"Critical translation error: {e}")
        return text
# def translate_pdf_direct(input_path: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
#     """Direct PDF translation using PyMuPDF to preserve layout exactly."""
#     try:
#         doc = pymupdf.open(input_path)
#         for page in doc:
#             blocks = page.get_text("blocks")  # Extract text blocks with bounding boxes
#             for block in blocks:
#                 bbox = block[:4]  # Bounding box (x0, y0, x1, y1)
#                 original_text = block[4].strip()
#                 if original_text:
#                     # Translate the text
#                     translated_text = translate_with_rules(original_text, target_lang, max_retries, app_state, kb_en_to_zh)
                    
#                     # Redact (hide) original text with white fill
#                     page.draw_rect(bbox, color=None, fill=(1, 1, 1))  # White background
                    
#                     # Insert translated text in the same box (auto-fits with suitable font)
#                     page.insert_htmlbox(bbox, translated_text, css="* {font-family: sans-serif;}", archive=None)
        
#         output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
#         doc.save(output_path)
#         doc.close()
#         return output_path
#     except Exception as e:
#         logger.error(f"Direct PDF translation failed: {e}")
#         return None  # Fallback to original if needed
import pymupdf
from typing import Tuple

def _fit_text_to_bbox(text: str, bbox: pymupdf.Rect, fontname: str = "helv") -> Tuple[float, str]:
    """Return (font_size, fitted_text) that fits in bbox"""
    width = bbox.width
    height = bbox.height
    max_lines = int(height / 12)  # rough max lines

    # Try font sizes from 6 to 14
    for fs in range(14, 5, -1):
        font = pymupdf.Font(fontname)
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if font.text_length(test_line, fontsize=fs) <= width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                if font.text_length(word, fontsize=fs) > width:
                    # Word too long — split it
                    while word:
                        part = word[:int(width / fs * 1.5)]
                        if font.text_length(part, fontsize=fs) <= width:
                            lines.append(part)
                            word = word[len(part):]
                        else:
                            break
                    current_line = ""
                    continue
        if current_line:
            lines.append(current_line)

        if len(lines) <= max_lines:
            return fs, "\n".join(lines)
    
    # If nothing fits, return smallest
    return 6, text[:100] + "..."  # truncate as last resort

def translate_pdf_direct(input_path: str, target_lang: str, max_retries: int = 2, app_state=None, kb_en_to_zh=None) -> str:
    try:
        pymupdf.TOOLS.set_small_glyph_heights(True)
        doc = pymupdf.open(input_path)
        page_texts = []  # Store all text per page

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            current_page_text = ""

            for block in blocks:
                if "lines" not in block:
                    continue
                bbox = pymupdf.Rect(block["bbox"])
                original_text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()

                if not original_text or len(original_text) < 3:
                    continue

                translated_text = translate_with_rules(original_text, target_lang, max_retries, app_state, kb_en_to_zh)

                # Use first span's font
                fontname = block["lines"][0]["spans"][0]["font"].lower() if block["lines"] else "helv"

                # Fit text
                font_size, fitted_text = _fit_text_to_bbox(translated_text, bbox, fontname)

                # Redact + Insert
                page.add_redact_annot(bbox, fill=(1,1,1))
                page.apply_redactions()
                page.insert_textbox(
                    bbox, fitted_text,
                    fontname=fontname,
                    fontsize=font_size,
                    lineheight=1.0,
                    align=0
                )

                current_page_text += fitted_text + " "

            page_texts.append(current_page_text.strip())
            page.clean_contents()

        # === REMOVE BLANK PAGES ===
        for page_num in reversed(range(len(doc))):
            page = doc[page_num]
            if len(page.get_text().strip()) < 50:  # almost empty
                doc.delete_page(page_num)

        output_path = input_path.replace(".pdf", f"_translated_{target_lang}.pdf")
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return output_path

    except Exception as e:
        logger.error(f"PDF translation failed: {e}")
        return None
# def cleanup_docx(doc: docx.Document):
#     """Post-translation cleanup: Remove empty paragraphs to prevent blank pages."""
#     for para in list(doc.paragraphs):          # Use list to avoid modification issues
#         if not para.text.strip():              # Check if paragraph is empty (no text)
#             para._element.getparent().remove(para._element)   # Remove it safely
    
#     for section in doc.sections:
#         for para in list(section.header.paragraphs + section.footer.paragraphs):
#             if not para.text.strip():
#                 para._element.getparent().remove(para._element)
def cleanup_docx(doc: docx.Document):
    from docx.shared import Pt

    # Remove only excessive consecutive empty paragraphs (preserve intentional blank lines)
    # Keep at least one empty paragraph between non-empty paragraphs for spacing
    paragraphs = list(doc.paragraphs)
    if not paragraphs:
        return
    
    # Track consecutive empty paragraphs
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        if not para.text.strip():
            # Count consecutive empty paragraphs
            empty_count = 1
            j = i + 1
            while j < len(paragraphs) and not paragraphs[j].text.strip():
                empty_count += 1
                j += 1
            
            # If more than 2 consecutive empty paragraphs, remove excess (keep max 2 for spacing)
            if empty_count > 2:
                for k in range(i + 2, j):
                    p = paragraphs[k]._element
                    p.getparent().remove(p)
                # Update paragraphs list after removal
                paragraphs = list(doc.paragraphs)
                i += 2  # Skip the kept empty paragraphs
            else:
                i = j
        else:
            i += 1

    # Optional: Shrink font slightly if needed (but preserve spacing)
    # Only adjust font size, do NOT modify paragraph spacing
    for para in doc.paragraphs:
        if para.text.strip():
            for run in para.runs:
                if run.font.size:
                    run.font.size = Pt(max(6, run.font.size.pt - 1))
            # PRESERVE original paragraph spacing - do not reset to zero
            # Original spacing is preserved by translate_docx_preserve_exact_formatting()
def translate_run_text(original_text: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    if not original_text or not original_text.strip():
        return original_text
    kb_hit = kb_lookup_fuzzy(kb_en_to_zh, original_text)
    if kb_hit:
        return kb_hit
    preserved = {}
    tmp, preserved_time = preserve_time_tokens(original_text)
    tmp, preserved_format = preserve_formatting_elements(tmp)
    preserved.update(preserved_time)
    preserved.update(preserved_format)
    translated = translate_text_enhanced(tmp, target_lang, max_retries=max_retries, app_state=app_state)
    translated = restore_formatting_elements(translated, preserved_format)
    translated = restore_time_tokens(translated, preserved_time)
    # Restore capitalization patterns from original text
    translated = preserve_and_restore_capitalization(original_text, translated)
    return translated

def replace_text_preserve_runs(paragraph: docx.text.paragraph.Paragraph, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None):
    if not paragraph.text.strip():
        return
    runs = paragraph.runs
    if not runs:
        return
    if len(runs) == 1:
        run = runs[0]
        run.text = translate_run_text(run.text, target_lang, max_retries, app_state, kb_en_to_zh)
        return
    new_texts = []
    run_groups = []
    buffer = ""
    buffer_runs = []
    for run in runs:
        text = run.text or ""
        if len(text.strip()) == 0:
            if buffer:
                run_groups.append((buffer, buffer_runs))
                buffer = ""
                buffer_runs = []
            run_groups.append((text, [run]))
            continue
        if len(text) < 6:
            buffer += text
            buffer_runs.append(run)
        else:
            if buffer:
                buffer += text
                buffer_runs.append(run)
                run_groups.append((buffer, buffer_runs))
                buffer = ""
                buffer_runs = []
            else:
                run_groups.append((text, [run]))
    if buffer:
        run_groups.append((buffer, buffer_runs))
    for group_text, group_runs in run_groups:
        translated = translate_run_text(group_text, target_lang, max_retries, app_state, kb_en_to_zh)
        first_run = group_runs[0]
        first_run.text = translated
        for rr in group_runs[1:]:
            rr.text = ""

def translate_text_batch(texts: List[str], target_lang: str, src_lang: str = 'en', max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None, batch_size: int = 8) -> List[str]:
    """
    Batch translate multiple texts in parallel for better performance.
    Uses model's batch processing capabilities and processes batches in parallel.
    
    Args:
        texts: List of texts to translate
        target_lang: Target language code
        src_lang: Source language code
        max_retries: Maximum retry attempts
        app_state: Application state with model and tokenizer
        batch_size: Number of texts to process in a single batch
        
    Returns:
        List of translated texts in the same order as input
    """
    if not texts:
        return []
    
    import time
    batch_start = time.time()
    logger.info(f"🔄 [BATCH] Starting batch translation of {len(texts)} texts to {target_lang} (batch_size={batch_size})")
    
    # Filter empty texts and track indices
    non_empty_texts = []
    indices = []
    for idx, text in enumerate(texts):
        if text and text.strip():
            non_empty_texts.append(text)
            indices.append(idx)
        else:
            indices.append(-1)  # Mark as empty
    
    if not non_empty_texts:
        return texts  # All empty
    
    # Process texts in batches
    translated_results = [None] * len(non_empty_texts)
    total_batches = (len(non_empty_texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(non_empty_texts), batch_size):
        batch_texts = non_empty_texts[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        try:
            logger.info(f"   📦 [BATCH] Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            batch_process_start = time.time()
            
            # Prepare texts for batch processing
            processed_batch = []
            preserved_data = []
            
            for text in batch_texts:
                cleaned = clean_text(text)
                processed, preserved_elements = preserve_formatting_elements(cleaned)
                processed, preserved_time = preserve_time_tokens(processed)
                
                # Check knowledge base for Chinese translations (if applicable)
                if target_lang == "zh" and src_lang == "en" and kb_en_to_zh:
                    key = re.sub(r"\s+", " ", processed.strip().lower())
                    kb_hit = kb_lookup_fuzzy(kb_en_to_zh, key)
                    if kb_hit:
                        # Use KB lookup result
                        preserved_data.append({
                            'elements': preserved_elements,
                            'time': preserved_time,
                            'kb_translation': kb_hit
                        })
                        processed_batch.append(None)  # Mark as KB translation
                        continue
                
                processed_batch.append(processed)
                preserved_data.append({
                    'elements': preserved_elements,
                    'time': preserved_time,
                    'kb_translation': None
                })
            
            # Separate KB translations from model translations
            texts_to_translate = [t for t in processed_batch if t is not None]
            kb_indices = [i for i, t in enumerate(processed_batch) if t is None]
            translate_indices = [i for i, t in enumerate(processed_batch) if t is not None]
            
            # Process KB translations
            for kb_idx in kb_indices:
                preserved = preserved_data[kb_idx]
                if preserved['kb_translation']:
                    original_text = batch_texts[kb_idx]  # Get original text for capitalization restoration
                    final = restore_formatting_elements(preserved['kb_translation'], preserved['elements'])
                    final = restore_time_tokens(final, preserved['time'])
                    # Restore capitalization patterns from original text
                    final = preserve_and_restore_capitalization(original_text, final)
                    translated_results[batch_idx + kb_idx] = final
            
            # Batch translate remaining texts if any
            if texts_to_translate:
                app_state.tokenizer.src_lang = src_lang
                encoded = app_state.tokenizer(
                    texts_to_translate,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(app_state.device)
                
                # Batch generate translations
                with torch.no_grad():
                    target_lang_code = get_language_code(target_lang)
                    forced_bos_token_id = app_state.tokenizer.lang_code_to_id.get(target_lang_code, None)
                    
                    if forced_bos_token_id is None:
                        generated_tokens = app_state.model.generate(
                            **encoded,
                            max_length=1024,
                            num_beams=3,
                            length_penalty=0.8,
                            early_stopping=True,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.2
                        )
                    else:
                        generated_tokens = app_state.model.generate(
                            **encoded,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=1024,
                            num_beams=3,
                            length_penalty=0.8,
                            early_stopping=True,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            repetition_penalty=1.2
                        )
                
                # Decode batch results
                batch_translations = app_state.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                # Restore formatting for each translation
                for trans_idx, (translated, orig_idx) in enumerate(zip(batch_translations, translate_indices)):
                    preserved = preserved_data[orig_idx]
                    original_text = batch_texts[orig_idx]  # Get original text for capitalization restoration
                    final = restore_formatting_elements(translated, preserved['elements'])
                    final = restore_time_tokens(final, preserved['time'])
                    # Restore capitalization patterns from original text
                    final = preserve_and_restore_capitalization(original_text, final)
                    translated_results[batch_idx + orig_idx] = final
            
            batch_process_time = time.time() - batch_process_start
            logger.info(f"   ✅ [BATCH] Batch {batch_num}/{total_batches} completed in {batch_process_time:.2f}s ({batch_process_time/len(batch_texts):.3f}s per text)")
            
        except Exception as e:
            logger.error(f"   ❌ [BATCH] Batch {batch_num} failed: {str(e)}, falling back to individual translation")
            # Fallback to individual translation for this batch
            for i, text in enumerate(batch_texts):
                try:
                    translated_results[batch_idx + i] = translate_text_enhanced(
                        text, target_lang=target_lang, src_lang=src_lang, 
                        max_retries=max_retries, app_state=app_state
                    )
                except Exception as e2:
                    logger.error(f"   ❌ [BATCH] Individual translation failed for text {batch_idx + i}: {str(e2)}")
                    translated_results[batch_idx + i] = text  # Fallback to original
    
    # Reconstruct results in original order
    result = [None] * len(texts)
    result_idx = 0
    for idx, orig_idx in enumerate(indices):
        if orig_idx == -1:
            result[idx] = texts[idx]  # Empty text
        else:
            result[idx] = translated_results[result_idx]
            result_idx += 1
    
    total_time = time.time() - batch_start
    logger.info(f"✅ [BATCH] Batch translation completed: {len(texts)} texts in {total_time:.2f}s (avg: {total_time/len(texts):.3f}s per text)")
    
    return result

def translate_with_rules(text: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    if not text or not text.strip():
        return text
    try:
        src_lang = detect(text)
        if src_lang.startswith('zh'):
            src_lang = 'zh'
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        src_lang = 'en'
    tl = target_lang.lower().strip()
    if src_lang == tl:
        return text
    cleaned_text = clean_text(text)
    processed_text, preserved_elements = preserve_formatting_elements(cleaned_text)
    processed_text, preserved_time = preserve_time_tokens(processed_text)
    if tl == "zh" and src_lang == "en":
        key = re.sub(r"\s+", " ", processed_text.strip().lower())
        kb_hit = kb_lookup_fuzzy(kb_en_to_zh, key)
        if kb_hit:
            restored = restore_formatting_elements(kb_hit, preserved_elements)
            restored = restore_time_tokens(restored, preserved_time)
            # Restore capitalization patterns from original text
            restored = preserve_and_restore_capitalization(text, restored)
            return restored
    translated = translate_text_enhanced(processed_text, target_lang=tl, src_lang=src_lang, max_retries=max_retries, app_state=app_state)
    restored = restore_formatting_elements(translated, preserved_elements)
    restored = restore_time_tokens(restored, preserved_time)
    # Restore capitalization patterns from original text
    restored = preserve_and_restore_capitalization(text, restored)
    return restored

def translate_pointwise(text: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    lines = text.splitlines()
    translated_lines = []
    for line in lines:
        if line.strip():
            translated_line = translate_with_rules(line, target_lang, max_retries, app_state, kb_en_to_zh)
            translated_lines.append(translated_line)
        else:
            translated_lines.append("")
    return "\n".join(translated_lines)

def copy_run_format(source_run: docx.text.run.Run, target_run: docx.text.run.Run):
    try:
        if source_run.font.name:
            target_run.font.name = source_run.font.name
        if source_run.font.size:
            target_run.font.size = source_run.font.size
        if source_run.bold is not None:
            target_run.bold = source_run.bold
        if source_run.italic is not None:
            target_run.italic = source_run.italic
        if source_run.underline is not None:
            target_run.underline = source_run.underline
        try:
            if source_run.font.color.rgb:
                target_run.font.color.rgb = source_run.font.color.rgb
        except:
            pass
        try:
            if source_run.font.highlight_color:
                target_run.font.highlight_color = source_run.font.highlight_color
        except:
            pass
    except Exception as e:
        logger.debug(f"Could not copy some formatting: {e}")

def save_paragraph_format(paragraph: docx.text.paragraph.Paragraph) -> dict:
    """Save all paragraph format attributes for later restoration."""
    fmt = paragraph.paragraph_format
    return {
        'style': paragraph.style,
        'alignment': paragraph.alignment,
        'space_before': fmt.space_before,
        'space_after': fmt.space_after,
        'line_spacing': fmt.line_spacing,
        'line_spacing_rule': fmt.line_spacing_rule,
        'first_line_indent': fmt.first_line_indent,
        'left_indent': fmt.left_indent,
        'right_indent': fmt.right_indent,
        'keep_together': fmt.keep_together,
        'keep_with_next': fmt.keep_with_next,
        'page_break_before': fmt.page_break_before,
        'widow_control': fmt.widow_control,
    }

def restore_paragraph_format(paragraph: docx.text.paragraph.Paragraph, saved_format: dict):
    """Restore all paragraph format attributes from saved format."""
    try:
        if 'style' in saved_format:
            paragraph.style = saved_format['style']
        if 'alignment' in saved_format and saved_format['alignment'] is not None:
            paragraph.alignment = saved_format['alignment']
        
        fmt = paragraph.paragraph_format
        if 'space_before' in saved_format and saved_format['space_before'] is not None:
            fmt.space_before = saved_format['space_before']
        if 'space_after' in saved_format and saved_format['space_after'] is not None:
            fmt.space_after = saved_format['space_after']
        if 'line_spacing' in saved_format and saved_format['line_spacing'] is not None:
            fmt.line_spacing = saved_format['line_spacing']
        if 'line_spacing_rule' in saved_format and saved_format['line_spacing_rule'] is not None:
            fmt.line_spacing_rule = saved_format['line_spacing_rule']
        if 'first_line_indent' in saved_format and saved_format['first_line_indent'] is not None:
            fmt.first_line_indent = saved_format['first_line_indent']
        if 'left_indent' in saved_format and saved_format['left_indent'] is not None:
            fmt.left_indent = saved_format['left_indent']
        if 'right_indent' in saved_format and saved_format['right_indent'] is not None:
            fmt.right_indent = saved_format['right_indent']
        if 'keep_together' in saved_format and saved_format['keep_together'] is not None:
            fmt.keep_together = saved_format['keep_together']
        if 'keep_with_next' in saved_format and saved_format['keep_with_next'] is not None:
            fmt.keep_with_next = saved_format['keep_with_next']
        if 'page_break_before' in saved_format and saved_format['page_break_before'] is not None:
            fmt.page_break_before = saved_format['page_break_before']
        if 'widow_control' in saved_format and saved_format['widow_control'] is not None:
            fmt.widow_control = saved_format['widow_control']
    except Exception as e:
        logger.debug(f"Could not restore some paragraph formatting: {e}")

def translate_docx_preserve_exact_formatting(input_path: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    try:
        doc = docx.Document(input_path)
        # Process main document paragraphs - preserve ALL formatting attributes
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                saved_format = save_paragraph_format(paragraph)
                replace_text_preserve_runs(paragraph, target_lang, max_retries, app_state, kb_en_to_zh)
                restore_paragraph_format(paragraph, saved_format)
        
        # Process table cells - preserve ALL formatting attributes
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        if paragraph.text.strip():
                            saved_format = save_paragraph_format(paragraph)
                            replace_text_preserve_runs(paragraph, target_lang, max_retries, app_state, kb_en_to_zh)
                            restore_paragraph_format(paragraph, saved_format)
        
        # Process headers - preserve ALL formatting attributes
        for section in doc.sections:
            try:
                for paragraph in section.header.paragraphs:
                    if paragraph.text.strip():
                        saved_format = save_paragraph_format(paragraph)
                        replace_text_preserve_runs(paragraph, target_lang, max_retries, app_state, kb_en_to_zh)
                        restore_paragraph_format(paragraph, saved_format)
            except Exception as e:
                logger.debug(f"Could not process header: {e}")
            # Process footers - preserve ALL formatting attributes
            try:
                for paragraph in section.footer.paragraphs:
                    if paragraph.text.strip():
                        saved_format = save_paragraph_format(paragraph)
                        replace_text_preserve_runs(paragraph, target_lang, max_retries, app_state, kb_en_to_zh)
                        restore_paragraph_format(paragraph, saved_format)
            except Exception as e:
                logger.debug(f"Could not process footer: {e}")
        output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
        doc.save(output_path)
        # After save/write:
        doc = docx.Document(output_path)  # Reload
        cleanup_docx(doc)
        doc.save(output_path)  # Re-save cleaned # Save again after cleanup
        return output_path
    except Exception as e:
        logger.error(f"DOCX formatting preservation failed: {e}")
        return translate_docx_xml_level(input_path, target_lang, max_retries, app_state, kb_en_to_zh)

def translate_docx_xml_level(input_path: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    import time
    start_time = time.time()
    logger.info(f"📄 [TRANSLATION] Starting DOCX XML translation: {os.path.basename(input_path)} -> {target_lang}")
    
    try:
        load_start = time.time()
        # Read zipfile and extract document.xml
        with zipfile.ZipFile(input_path, 'r') as docx_zip:
            document_xml = docx_zip.read('word/document.xml')
            # Store zipfile info for later use
            zipfile_info = docx_zip.infolist()
        
        root = ET.fromstring(document_xml)
        load_time = time.time() - load_start
        logger.info(f"   ⏱️  [TRANSLATION] File loaded in {load_time:.3f}s")
        
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text_elements = root.findall('.//w:t', namespaces)
        
        # Collect all text elements with their indices
        elements_to_translate = []
        element_indices = []
        for idx, t_elem in enumerate(text_elements):
            if t_elem.text and t_elem.text.strip():
                elements_to_translate.append(t_elem)
                element_indices.append(idx)
        
        total_elements = len(elements_to_translate)
        logger.info(f"   📊 [TRANSLATION] Found {total_elements} text elements to translate")
        
        if total_elements == 0:
            logger.info(f"   ⚠️  [TRANSLATION] No text elements found to translate")
            # No translation needed, return original
            output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
            with zipfile.ZipFile(input_path, 'r') as source_zip:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                    for item in source_zip.infolist():
                        new_zip.writestr(item, source_zip.read(item.filename))
            return output_path
        
        # Extract all texts for batch processing
        texts_to_translate = [elem.text for elem in elements_to_translate]
        
        # Detect source language once for all texts (use first non-empty text)
        translation_start = time.time()
        try:
            sample_text = texts_to_translate[0] if texts_to_translate else ""
            src_lang = detect(sample_text[:500]) if sample_text else 'en'
            if src_lang.startswith('zh'):
                src_lang = 'zh'
            logger.info(f"   🌍 [TRANSLATION] Detected source language: {src_lang} (using sample from first element)")
        except Exception as e:
            logger.warning(f"   ⚠️  [TRANSLATION] Language detection failed: {e}, defaulting to 'en'")
            src_lang = 'en'
        
        # Check if translation is needed
        tl = target_lang.lower().strip()
        if src_lang == tl:
            logger.info(f"   ℹ️  [TRANSLATION] Source and target languages are the same ({src_lang}), skipping translation")
            # Copy original file
            output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
            with zipfile.ZipFile(input_path, 'r') as source_zip:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                    for item in source_zip.infolist():
                        new_zip.writestr(item, source_zip.read(item.filename))
            return output_path
        
        # Use batch translation for better performance
        logger.info(f"   🚀 [TRANSLATION] Using BATCH translation mode (parallel processing enabled)")
        batch_size = 8  # Process 8 texts at a time in each batch
        translated_texts = translate_text_batch(
            texts_to_translate, 
            target_lang=tl, 
            src_lang=src_lang, 
            max_retries=max_retries, 
            app_state=app_state,
            kb_en_to_zh=kb_en_to_zh,
            batch_size=batch_size
        )
        
        # Apply translations back to XML elements
        for elem, translated_text in zip(elements_to_translate, translated_texts):
            elem.text = translated_text
        
        translation_time = time.time() - translation_start
        logger.info(f"   ✅ [TRANSLATION] All {total_elements} elements translated in {translation_time:.2f}s (avg: {translation_time/max(total_elements,1):.3f}s per element)")
        
        # Save the translated document
        save_start = time.time()
        output_buffer = BytesIO()
        with zipfile.ZipFile(output_buffer, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            # Re-read original zipfile to get all files
            with zipfile.ZipFile(input_path, 'r') as source_zip:
                for item in source_zip.infolist():
                    if item.filename != 'word/document.xml':
                        new_zip.writestr(item, source_zip.read(item.filename))
            new_document_xml = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            new_zip.writestr('word/document.xml', new_document_xml)
        
        output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
        with open(output_path, 'wb') as f:
            f.write(output_buffer.getvalue())
        
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        logger.info(f"   💾 [TRANSLATION] File saved in {save_time:.3f}s")
        logger.info(f"   ✅ [TRANSLATION] Total translation time: {total_time:.2f}s for {os.path.basename(input_path)}")
        return output_path
    except Exception as e:
        logger.error(f"XML-level translation failed: {e}")
        return None

def translate_pptx_preserve(input_path: str, target_lang: str, max_retries: int = 2, app_state: AppState = None, kb_en_to_zh: Dict[str, str] = None) -> str:
    prs = Presentation(input_path)
    def xlate(s: str) -> str:
        return translate_with_rules(s, target_lang, max_retries, app_state, kb_en_to_zh) if s and s.strip() else s
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text_frame") and shape.text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if run.text and run.text.strip():
                            run.text = xlate(run.text)
    output_path = os.path.join(os.path.dirname(input_path), f"translated_{target_lang}_{os.path.basename(input_path)}")
    prs.save(output_path)
    return output_path