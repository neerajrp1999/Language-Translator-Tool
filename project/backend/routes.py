from fastapi import APIRouter, UploadFile, Form, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import uuid
import os
import tempfile
import logging
import json
import re
from functools import partial
from langdetect import detect
from .models import TextTranslationRequest, ChatRequest
from .translation import (
    translate_with_rules,
    translate_docx_preserve_exact_formatting,
    translate_docx_xml_level,
    translate_pptx_preserve,
    translate_text_enhanced,
    translate_pointwise
)
from .file_handlers import (
    extract_text_from_file,
    convert_pdf_to_docx,
    convert_docx_to_pdf,
    read_docx_enhanced,
    read_pdf_enhanced,
    read_pptx_enhanced,
    validate_formatting_preservation
)
from .evaluation import evaluate_translation_with_llama
from .ollama import get_llama_response_async
from .utils import mock_rag_lookup, mock_db_update, clean_text, highlight_deltas, get_language_code
from .config import LANG_OPTIONS, evaluation_config, ENABLE_PARALLEL, MAX_RETRIES, USE_XML_METHOD
from .state import AppState, get_app_state  # Import get_app_state from state.py
from .translation import (
    translate_with_rules,
    translate_docx_preserve_exact_formatting,
    translate_docx_xml_level,
    translate_pptx_preserve,
    translate_text_enhanced,
    translate_pointwise,
    translate_pdf_direct  # New import
)
logger = logging.getLogger(__name__)

router = APIRouter()

async def process_translation_job_from_path(filename: str, source_path: str, target_lang: str, app_state: AppState) -> Dict[str, Any]:
    """
    Helper function to process translation job from file path.
    Matches temp.py implementation.
    """
    ext = os.path.splitext(filename)[1].lower()
    input_docx_path = None
    translated_path = None
    output_path = None
    download_name = f"translated_{target_lang}_{filename}"
    mime = "application/octet-stream"
    
    loop = asyncio.get_event_loop()
    
    try:
        if ext not in [".docx", ".pdf", ".pptx"]:
            raise ValueError("Unsupported file type.")
            
        if ext == ".docx":
            input_path = source_path
            # Run translation in executor to avoid blocking
            translate_func = translate_docx_xml_level if USE_XML_METHOD else translate_docx_preserve_exact_formatting
            translated_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(translate_func, input_path, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
            )
            if not translated_path:
                raise ValueError("Translation failed")
            output_path = translated_path
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
        elif ext == ".pdf":
            # Try direct PDF translation first (preserves layout, no blank pages)
            translated_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(translate_pdf_direct, source_path, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
            )
            
            if not translated_path:
                # Direct method failed — log it and fall back to old DOCX method
                logger.warning("Direct PDF translation failed. Falling back to PDF→DOCX→PDF method.")
                
                # === OLD FALLBACK PATH (unchanged) ===
                input_docx_path = await loop.run_in_executor(
                    app_state.file_io_executor,
                    partial(convert_pdf_to_docx, source_path)
                )
                if not input_docx_path:
                    raise ValueError("PDF to DOCX conversion failed")
                
                translate_func = translate_docx_xml_level if USE_XML_METHOD else translate_docx_preserve_exact_formatting
                translated_path = await loop.run_in_executor(
                    app_state.translation_executor,
                    partial(translate_func, input_docx_path, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
                )
                output_path_pdf = await loop.run_in_executor(
                    app_state.file_io_executor,
                    partial(convert_docx_to_pdf, translated_path)
                )
                output_path = output_path_pdf or translated_path
                download_name = f"translated_{target_lang}_{os.path.splitext(filename)[0]}.pdf" if output_path_pdf else f"translated_{target_lang}_{os.path.splitext(filename)[0]}.docx"
                mime = "application/pdf" if output_path_pdf else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                # Direct translation succeeded → use result
                output_path = translated_path
                download_name = f"translated_{target_lang}_{os.path.basename(filename)}"
                mime = "application/pdf"      
        elif ext == ".pptx":
            # PPTX translation in executor
            output_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(translate_pptx_preserve, source_path, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
            )

            mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            
        file_id = str(uuid.uuid4())
        app_state.translated_files[file_id] = {"path": output_path, "name": download_name, "mime": mime}
        
        # Read file content in executor for language detection
        source_text_sample = ""
        if ext == ".docx":
            source_text_sample = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_docx_enhanced, source_path)
            )
        elif ext == ".pdf":
            source_text_sample = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_pdf_enhanced, source_path)
            )
        else:
            source_text_sample = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_pptx_enhanced, source_path)
            )
            
        # Language detection in executor
        source_lang_key = 'unknown'
        if source_text_sample:
            try:
                source_lang_key = await loop.run_in_executor(
                    app_state.translation_executor,
                    partial(detect, source_text_sample[:500].strip())
                )
            except:
                source_lang_key = 'unknown'
        
        return {
            "source_filename": filename,
            "target_language": target_lang,
            "source_language": source_lang_key,
            "status": "Success",
            "download_id": file_id
        }
        
    except Exception as e:
        logger.error(f"Translation job failed: {str(e)}")
        return {
            "source_filename": filename,
            "target_language": target_lang,
            "status": "Failed",
            "detail": str(e)
        }

@router.post("/translate_text")
async def translate_text(request: TextTranslationRequest, app_state: AppState = Depends(get_app_state)):
    target_lang_key = request.target_language.lower().strip()
    if target_lang_key not in LANG_OPTIONS:
        raise HTTPException(400, f"Invalid target language: {target_lang_key}")
    target_lang_code = LANG_OPTIONS[target_lang_key]
    
    source_lang_code = None
    if request.source_language:
        source_lang_key = request.source_language.lower().strip()
        if source_lang_key not in LANG_OPTIONS:
            raise HTTPException(400, f"Invalid source language: {source_lang_key}")
        source_lang_code = LANG_OPTIONS[source_lang_key]
    
    if not request.text or not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    
    loop = asyncio.get_event_loop()
    
    try:
        if not source_lang_code:
            detected_lang = await loop.run_in_executor(
                app_state.file_io_executor,
                partial(detect, request.text[:evaluation_config.text_limits['language_detection_sample']])
            )

            source_lang_code = detected_lang if detected_lang in LANG_OPTIONS.values() else 'en'
            logger.info(f"Detected source language: {source_lang_code}")
        
        if source_lang_code == target_lang_code:
            return JSONResponse({
                "translated_text": request.text,
                "source_language": next((k for k, v in LANG_OPTIONS.items() if v == source_lang_code), source_lang_code),
                "target_language": request.target_language,
                "message": "Source and target languages are the same. No translation needed."
            })
        
        translated_text = await loop.run_in_executor(
            app_state.translation_executor,
            partial(translate_with_rules, request.text, target_lang_code, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
        )
        
        source_lang_name = next((k for k, v in LANG_OPTIONS.items() if v == source_lang_code), source_lang_code)
        target_lang_name = next((k for k, v in LANG_OPTIONS.items() if v == target_lang_code), target_lang_code)
        
        return JSONResponse({
            "translated_text": translated_text,
            "source_language": source_lang_name,
            "target_language": target_lang_name,
            "source_text_length": len(request.text),
            "translated_text_length": len(translated_text),
            "message": "Text translation complete."
        })
        
    except Exception as e:
        logger.error(f"Text translation failed: {str(e)}")
        raise HTTPException(500, f"Text translation failed: {str(e)}")

@router.post("/translate")
async def translate_file(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    app_state: AppState = Depends(get_app_state)
):
    """
    Translate a single file (DOCX, PDF, or PPTX) to target language.
    Returns file_id for download and preview text.
    Matches temp.py behavior exactly.
    """
    target_lang_key = target_language.lower().strip()
    if target_lang_key not in LANG_OPTIONS:
        raise HTTPException(400, f"Invalid target language: {target_lang_key}. Supported: {', '.join(LANG_OPTIONS.keys())}")
    target_lang = LANG_OPTIONS[target_lang_key]
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".docx", ".pdf", ".pptx"]:
        raise HTTPException(400, "Unsupported file type. Supported: docx, pdf, pptx")
    
    tmp_path = None
    loop = asyncio.get_event_loop()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Process translation synchronously (like temp.py)
        result = await process_translation_job_from_path(
            filename=file.filename,
            source_path=tmp_path,
            target_lang=target_lang,
            app_state=app_state
        )
        
        if result['status'] == 'Failed':
            raise HTTPException(500, f"Translation failed: {result['detail']}")
        
        data = app_state.translated_files[result['download_id']]
        output_path = data['path']
        
        # Run file reading in executor to avoid blocking (consistency with temp.py)
        preview_text = ""
        if ext == ".docx":
            preview_text = await loop.run_in_executor(
                app_state.file_io_executor,
                partial(read_docx_enhanced, output_path)
            )

        elif ext == ".pdf":
            if data['mime'] == "application/pdf":
                preview_text = await loop.run_in_executor(
                    app_state.file_io_executor,
                    partial(read_docx_enhanced, output_path)
                )

            else:
                preview_text = await loop.run_in_executor(
                    app_state.file_io_executor,
                    partial(read_docx_enhanced, output_path)
                )

        elif ext == ".pptx":
            preview_text = await loop.run_in_executor(
                app_state.file_io_executor,
                partial(read_docx_enhanced, output_path)
            )

        return JSONResponse({
            "file_id": result['download_id'],
            "source_language": result.get('source_language', 'unknown'),
            "target_language": target_lang_key,
            "preview": preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text,
            "message": "Translation complete. Use /download/{file_id} to download and /preview/{file_id} for full preview text."
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File translation failed: {str(e)}")
        raise HTTPException(500, f"Translation failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@router.post("/translate_files_batch")
async def translate_files_batch(
    files: List[UploadFile] = File(...),
    target_language: str = Form(...),
    enable_evaluation: bool = Form(True),
    background_tasks: BackgroundTasks = None,
    app_state: AppState = Depends(get_app_state)
):
    target_lang_key = target_language.lower().strip()
    if target_lang_key not in LANG_OPTIONS:
        raise HTTPException(400, f"Invalid target language: {target_lang_key}")
    target_lang_code = LANG_OPTIONS[target_lang_key]

    batch_job_id = str(uuid.uuid4())
    results = []
    temp_files = []

    try:
        for file in files:
            _, ext = os.path.splitext(file.filename.lower())
            if ext not in [".docx", ".pdf", ".pptx", ".txt"]:
                results.append({
                    "original_filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported file type: {ext}"
                })
                continue

            job_id = str(uuid.uuid4())
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_path = temp_file.name
                temp_files.append(temp_path)
                content = await file.read()
                temp_file.write(content)

            try:
                if ext == ".pdf":
                    docx_path = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(convert_pdf_to_docx, temp_path)
                    )
                    if not docx_path:
                        raise HTTPException(500, "PDF to DOCX conversion failed")
                    translated_path = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(
                            translate_docx_preserve_exact_formatting if not USE_XML_METHOD else translate_docx_xml_level,
                            docx_path,
                            target_lang_code,
                            MAX_RETRIES,
                            app_state,
                            app_state.kb_en_to_zh
                        )
                    )
                    final_path = translated_path
                    if ext == ".pdf":
                        pdf_path = await asyncio.get_event_loop().run_in_executor(
                            app_state.translation_executor,
                            partial(convert_docx_to_pdf, translated_path)
                        )
                        if pdf_path:
                            final_path = pdf_path
                            if os.path.exists(translated_path):
                                os.unlink(translated_path)
                        else:
                            logger.warning("PDF conversion failed, returning DOCX")
                elif ext == ".docx":
                    final_path = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(
                            translate_docx_preserve_exact_formatting if not USE_XML_METHOD else translate_docx_xml_level,
                            temp_path,
                            target_lang_code,
                            MAX_RETRIES,
                            app_state,
                            app_state.kb_en_to_zh
                        )
                    )
                elif ext == ".pptx":
                    final_path = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(translate_pptx_preserve, temp_path, target_lang_code, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
                    )
                else:  # .txt
                    content = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(extract_text_from_file, temp_path, ext)
                    )
                    translated_text = await asyncio.get_event_loop().run_in_executor(
                        app_state.translation_executor,
                        partial(translate_with_rules, content, target_lang_code, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
                    )
                    final_path = os.path.join(tempfile.gettempdir(), f"translated_{job_id}{ext}")
                    with open(final_path, "w", encoding="utf-8") as f:
                        f.write(translated_text)

                if not final_path or not os.path.exists(final_path):
                    raise HTTPException(500, "Translation failed: Output file not generated")

                source_text = await asyncio.get_event_loop().run_in_executor(
                    app_state.translation_executor,
                    partial(extract_text_from_file, temp_path, ext)
                )
                translated_text = await asyncio.get_event_loop().run_in_executor(
                    app_state.translation_executor,
                    partial(extract_text_from_file, final_path, ext)
                )

                app_state.translated_files[job_id] = {
                    "path": final_path,
                    "original_filename": file.filename,
                    "translated_filename": f"translated_{target_lang_code}_{file.filename}",
                    "target_language": target_lang_code,
                    "timestamp": datetime.utcnow().isoformat(),
                    "original_path": temp_path
                }

                app_state.jobs[job_id] = {
                    "status": "completed",
                    "original_filename": file.filename,
                    "translated_filename": f"translated_{target_lang_code}_{file.filename}",
                    "target_language": target_lang_code,
                    "timestamp": datetime.utcnow().isoformat(),
                    "evaluation_id": None,
                    "batch_job_id": batch_job_id
                }

                if enable_evaluation:
                    try:
                        source_lang = await asyncio.get_event_loop().run_in_executor(
                            app_state.translation_executor,
                            partial(detect, source_text[:evaluation_config.text_limits['language_detection_sample']])
                        )
                        source_lang = source_lang if source_lang in LANG_OPTIONS.values() else 'en'
                        evaluation_result = await evaluate_translation_with_llama(
                            source_text,
                            translated_text,
                            source_lang,
                            target_lang_code,
                            file.filename,
                            f"translated_{target_lang_code}_{file.filename}"
                        )
                        evaluation_id = evaluation_result["evaluation_id"]
                        app_state.evaluation_results[evaluation_id] = evaluation_result
                        app_state.jobs[job_id]["evaluation_id"] = evaluation_id
                    except Exception as e:
                        logger.error(f"Evaluation failed for job {job_id}: {str(e)}")
                        app_state.jobs[job_id]["evaluation_id"] = None

                results.append({
                    "job_id": job_id,
                    "status": "completed",
                    "original_filename": file.filename,
                    "translated_filename": f"translated_{target_lang_code}_{file.filename}",
                    "message": "File translation completed",
                    "evaluation_id": app_state.jobs[job_id]["evaluation_id"]
                })

            except Exception as e:
                logger.error(f"File translation failed for {file.filename}: {str(e)}")
                app_state.jobs[job_id] = {
                    "status": "failed",
                    "original_filename": file.filename,
                    "error": str(e),
                    "batch_job_id": batch_job_id
                }
                results.append({
                    "job_id": job_id,
                    "status": "failed",
                    "original_filename": file.filename,
                    "error": str(e)
                })

        return JSONResponse({
            "batch_job_id": batch_job_id,
            "results": results,
            "message": "Batch translation completed"
        })

    finally:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

@router.get("/download/{file_id}")
async def download_file(file_id: str, app_state: AppState = Depends(get_app_state)):
    """Download translated file by file_id. Matches temp.py."""
    if file_id not in app_state.translated_files:
        raise HTTPException(404, "File not found")
    data = app_state.translated_files[file_id]
    return FileResponse(data["path"], media_type=data["mime"], filename=data["name"])

@router.get("/preview/{file_id}")
async def preview_file(file_id: str, app_state: AppState = Depends(get_app_state)):
    """
    Get full preview text of a translated file.
    Runs file reading in executor to avoid blocking.
    Matches temp.py.
    """
    if file_id not in app_state.translated_files:
        raise HTTPException(404, "File not found")
    
    data = app_state.translated_files[file_id]
    ext = os.path.splitext(data["name"])[1].lower()
    
    loop = asyncio.get_event_loop()
    text = ""
    
    try:
        # Run file reading in executor to avoid blocking
        if ext == ".docx":
            text = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_docx_enhanced, data["path"])
            )
        elif ext == ".pdf":
            text = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_pdf_enhanced, data["path"])
            )
        elif ext == ".pptx":
            text = await loop.run_in_executor(
                app_state.translation_executor,
                partial(read_pptx_enhanced, data["path"])
            )
        else:
            text = ""
        
        return JSONResponse({
            "preview": text,
            "file_id": file_id,
            "filename": data["name"],
            "length": len(text)
        })
    except Exception as e:
        logger.error(f"Preview generation failed for {file_id}: {str(e)}")
        raise HTTPException(500, f"Failed to generate preview: {str(e)}")

@router.get("/status/{job_id}")
async def get_status(job_id: str, app_state: AppState = Depends(get_app_state)):
    job = app_state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Helper function to convert datetime to ISO string
    def convert_datetime(dt):
        if isinstance(dt, datetime):
            return dt.isoformat()
        return dt
    
    # Initialize the response dictionary
    updated_at = convert_datetime(job.get("updated_at", job.get("timestamp")))
    
    response = {
        "job_id": job_id,
        "filename": job.get("filename", job.get("original_filename")),
        "target_language": job["target_language"],
        "status": job["status"],
        "updated_at": updated_at,
    }
    
    # If the job is completed, include the download_id
    if job["status"] == "COMPLETED" and "download_id" in job:
        response["download_id"] = job["download_id"]
    
    # Include evaluation_id if present
    if "evaluation_id" in job:
        response["evaluation_id"] = job["evaluation_id"]
    
    # Include error if present
    if "error" in job:
        response["error"] = job["error"]
    
    # Include created_at if present
    if "created_at" in job:
        response["created_at"] = convert_datetime(job["created_at"])
    
    return JSONResponse(response)

@router.get("/download_async/{job_id}")
async def download_async(job_id: str, app_state: AppState = Depends(get_app_state)):
    job = app_state.jobs.get(job_id)
    if not job or job["status"] != "COMPLETED":
        raise HTTPException(404, "File not ready or job not found")
    
    # Use the path and mime from the job
    file_path = job.get("path")
    mime_type = job.get("mime", "application/octet-stream")
    filename = job.get("filename", "translated_file")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "Translated file not found on disk")
    
    return FileResponse(
        file_path,
        media_type=mime_type,
        filename=f"translated_{filename}"
    )

@router.get("/evaluation/{evaluation_id}")
async def get_evaluation(evaluation_id: str, app_state: AppState = Depends(get_app_state)):
    if evaluation_id not in app_state.evaluation_results:
        raise HTTPException(404, "Evaluation not found")
    
    return JSONResponse(app_state.evaluation_results[evaluation_id])

@router.get("/evaluation/config")
async def get_evaluation_config():
    return JSONResponse({
        "weights": evaluation_config.weights,
        "quality_thresholds": evaluation_config.quality_thresholds,
        "text_limits": evaluation_config.text_limits,
        "model_config": evaluation_config.model_config
    })

@router.post("/evaluation/retry/{job_id}")
async def retry_evaluation(job_id: str, app_state: AppState = Depends(get_app_state)):
    if job_id not in app_state.jobs:
        raise HTTPException(404, "Job not found")
    
    job = app_state.jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed")
    
    file_data = app_state.translated_files.get(job_id)
    if not file_data:
        raise HTTPException(404, "Translated file not found")
    
    try:
        source_path = file_data.get("original_path")
        translated_path = file_data["path"]
        _, ext = os.path.splitext(translated_path.lower())
        
        source_text = await asyncio.get_event_loop().run_in_executor(
            app_state.translation_executor,
            partial(extract_text_from_file, source_path, ext)
        )
        translated_text = await asyncio.get_event_loop().run_in_executor(
            app_state.translation_executor,
            partial(extract_text_from_file, translated_path, ext)
        )
        
        source_lang = await asyncio.get_event_loop().run_in_executor(
            app_state.translation_executor,
            partial(detect, source_text[:evaluation_config.text_limits['language_detection_sample']])
        )
        source_lang = source_lang if source_lang in LANG_OPTIONS.values() else 'en'
        
        evaluation_result = await evaluate_translation_with_llama(
            source_text,
            translated_text,
            source_lang,
            file_data["target_language"],
            file_data["original_filename"],
            file_data["translated_filename"]
        )
        
        evaluation_id = evaluation_result["evaluation_id"]
        app_state.evaluation_results[evaluation_id] = evaluation_result
        app_state.jobs[job_id]["evaluation_id"] = evaluation_id
        
        return JSONResponse({
            "job_id": job_id,
            "evaluation_id": evaluation_id,
            "message": "Evaluation retried successfully"
        })
    
    except Exception as e:
        logger.error(f"Evaluation retry failed for job {job_id}: {str(e)}")
        raise HTTPException(500, f"Evaluation retry failed: {str(e)}")

@router.post("/chat")
async def chat(request: ChatRequest, app_state: AppState = Depends(get_app_state)):
    """
    Conversational translation endpoint.
    Matches temp.py behavior exactly.
    """
    user_message = request.message
    target_lang_key = request.target_language.lower()
    if target_lang_key not in LANG_OPTIONS:
        raise HTTPException(
            400,
            f"Invalid target language: {target_lang_key}. Supported: {', '.join(LANG_OPTIONS.keys())}"
        )
    target_lang_code = LANG_OPTIONS[target_lang_key]
    
    loop = asyncio.get_event_loop()
    
    try:
        # Run language detection in executor (can be slow for large texts)
        source_lang_code = await loop.run_in_executor(
            app_state.translation_executor,
            partial(detect, user_message[:500])  # Limit text for faster detection
        )
        source_lang_full = next((name for name, code in LANG_OPTIONS.items() if code == source_lang_code), "the user's language").title()
    except Exception:
        source_lang_full = "the user's language"
        
    rag_context = mock_rag_lookup(user_message)
    system_prompt = (
        f"You are a helpful and polite conversational agent. Your goal is to formulate a full, "
        f"context-aware response to the user's message in the **{source_lang_full}** language. "
        f"Do NOT translate the response. If RAG context is provided, use it to ensure accurate terminology."
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"--- RAG Context (Mocked) ---\n{rag_context}\n"
        f"--- New User Message ---\n{user_message}\n"
        f"Agent: "
    )
    
    # Use async Ollama call to avoid blocking
    source_response = await get_llama_response_async(prompt)
    if not source_response:
        raise HTTPException(503, "Conversational service unavailable (Ollama error)")
    
    # Run translation in executor to avoid blocking
    final_translated_response = await loop.run_in_executor(
        app_state.translation_executor,
        partial(translate_with_rules, source_response, target_lang_code, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
    )
    
    if not final_translated_response:
        logger.warning("M2M100 translation failed. Using Llama's source response as a fallback.")
        final_translated_response = source_response
        
    mock_db_update()
    return JSONResponse({"response": final_translated_response})

async def run_translation_job_background(
    job_id: str,
    filename: str,
    source_path: str,
    target_lang: str,
    enable_evaluation: bool,
    app_state: AppState
):
    """Background task to process translation job asynchronously"""
    job_start_time = datetime.utcnow()
    logger.info("=" * 80)
    logger.info(f"🔄 [PARALLEL JOB] Job {job_id} STARTED at {job_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logger.info(f"   📄 File: {filename}")
    logger.info(f"   🌍 Target language: {target_lang}")
    logger.info(f"   📁 Source path: {source_path}")
    logger.info(f"   ⚙️ Evaluation: {enable_evaluation}")
    logger.info(f"   🧵 Running in ThreadPoolExecutor (parallel processing enabled)")
    logger.info("=" * 80)
    
    app_state.jobs[job_id]["status"] = "PROCESSING"
    app_state.jobs[job_id]["updated_at"] = datetime.utcnow()  # Stored as datetime, converted to string in get_status
    
    try:
        ext = os.path.splitext(filename)[1].lower()
        loop = asyncio.get_event_loop()
        
        # Translate based on file type
        logger.info(f"   🔄 [JOB {job_id}] Starting translation for file type: {ext}")
        if ext == ".pdf":
            logger.info(f"   📄 [JOB {job_id}] Converting PDF to DOCX...")
            docx_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(convert_pdf_to_docx, source_path)
            )
            if not docx_path:
                raise Exception("PDF to DOCX conversion failed")
            logger.info(f"   ✅ [JOB {job_id}] PDF converted to DOCX: {docx_path}")
            
            logger.info(f"   🌐 [JOB {job_id}] Translating DOCX to {target_lang}...")
            translated_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(
                    translate_docx_preserve_exact_formatting if not USE_XML_METHOD else translate_docx_xml_level,
                    docx_path,
                    target_lang,
                    MAX_RETRIES,
                    app_state,
                    app_state.kb_en_to_zh
                )
            )
            
            logger.info(f"   📄 [JOB {job_id}] Converting translated DOCX back to PDF...")
            pdf_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(convert_docx_to_pdf, translated_path)
            )
            
            if pdf_path:
                logger.info(f"   ✅ [JOB {job_id}] PDF conversion completed: {pdf_path}")
                final_path = pdf_path
                mime = "application/pdf"
                if os.path.exists(translated_path):
                    os.unlink(translated_path)
            else:
                logger.warning(f"   ⚠️ [JOB {job_id}] PDF conversion failed, returning DOCX instead")
                final_path = translated_path
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        elif ext == ".docx":
            logger.info(f"   🌐 [JOB {job_id}] Translating DOCX to {target_lang}...")
            final_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(
                    translate_docx_preserve_exact_formatting if not USE_XML_METHOD else translate_docx_xml_level,
                    source_path,
                    target_lang,
                    MAX_RETRIES,
                    app_state,
                    app_state.kb_en_to_zh
                )
            )
            logger.info(f"   ✅ [JOB {job_id}] DOCX translation completed: {final_path}")
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        elif ext == ".pptx":
            logger.info(f"   🌐 [JOB {job_id}] Translating PPTX to {target_lang}...")
            final_path = await loop.run_in_executor(
                app_state.translation_executor,
                partial(translate_pptx_preserve, source_path, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
            )
            logger.info(f"   ✅ [JOB {job_id}] PPTX translation completed: {final_path}")
            mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        
        else:  # .txt
            logger.info(f"   📝 [JOB {job_id}] Extracting text from {ext} file...")
            content = await loop.run_in_executor(
                app_state.translation_executor,
                partial(extract_text_from_file, source_path, ext)
            )
            logger.info(f"   🌐 [JOB {job_id}] Translating text to {target_lang}...")
            translated_text = await loop.run_in_executor(
                app_state.translation_executor,
                partial(translate_pointwise, content, target_lang, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
            )
            final_path = os.path.join(tempfile.gettempdir(), f"translated_{job_id}{ext}")
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            logger.info(f"   ✅ [JOB {job_id}] Text translation completed: {final_path}")
            mime = "text/plain"
        
        if not final_path or not os.path.exists(final_path):
            raise Exception("Translation failed: Output file not generated")
        
        # Store translated file
        file_id = str(uuid.uuid4())
        app_state.translated_files[file_id] = {
            "path": final_path,
            "name": f"translated_{target_lang}_{filename}",
            "mime": mime,
            "original_filename": filename,
            "translated_filename": f"translated_{target_lang}_{filename}",
            "target_language": target_lang,
            "timestamp": datetime.utcnow().isoformat(),
            "original_path": source_path
        }
        
        # Update job with completion info
        logger.info(f"   💾 [JOB {job_id}] Storing translated file...")
        app_state.jobs[job_id].update({
            "status": "COMPLETED",
            "path": final_path,
            "mime": mime,
            "download_id": file_id,
            "updated_at": datetime.utcnow()
        })
        logger.info(f"   ✅ [JOB {job_id}] File stored with download_id: {file_id}")
        
        # Run evaluation if enabled
        if enable_evaluation:
            logger.info(f"   📊 [JOB {job_id}] Starting evaluation...")
            try:
                source_text = await loop.run_in_executor(
                    app_state.translation_executor,
                    partial(extract_text_from_file, source_path, ext)
                )
                translated_text = await loop.run_in_executor(
                    app_state.translation_executor,
                    partial(extract_text_from_file, final_path, ext)
                )
                
                source_lang = await loop.run_in_executor(
                    app_state.translation_executor,
                    partial(detect, source_text[:evaluation_config.text_limits['language_detection_sample']])
                )
                source_lang = source_lang if source_lang in LANG_OPTIONS.values() else 'en'
                
                evaluation_result = await evaluate_translation_with_llama(
                    source_text,
                    translated_text,
                    source_lang,
                    target_lang,
                    filename,
                    f"translated_{target_lang}_{filename}"
                )
                
                evaluation_id = evaluation_result["evaluation_id"]
                app_state.evaluation_results[evaluation_id] = evaluation_result
                app_state.jobs[job_id]["evaluation_id"] = evaluation_id
                logger.info(f"   ✅ [JOB {job_id}] Evaluation completed: {evaluation_id}")
            except Exception as e:
                logger.error(f"   ❌ [JOB {job_id}] Evaluation failed: {str(e)}")
                app_state.jobs[job_id]["evaluation_id"] = None
        
        job_end_time = datetime.utcnow()
        job_duration = (job_end_time - job_start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"✅ [PARALLEL JOB] Job {job_id} COMPLETED successfully")
        logger.info(f"   📄 File: {filename}")
        logger.info(f"   🌍 Target language: {target_lang}")
        logger.info(f"   ⏱️  Duration: {job_duration:.2f} seconds")
        logger.info(f"   💾 Download ID: {file_id}")
        logger.info(f"   📁 Output path: {final_path}")
        logger.info(f"   🕐 Started: {job_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logger.info(f"   🕐 Ended: {job_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logger.info("=" * 80)
        
    except Exception as e:
        job_end_time = datetime.utcnow()
        job_duration = (job_end_time - job_start_time).total_seconds()
        logger.error("=" * 80)
        logger.error(f"❌ [PARALLEL JOB] Job {job_id} FAILED")
        logger.error(f"   📄 File: {filename}")
        logger.error(f"   🌍 Target language: {target_lang}")
        logger.error(f"   ⏱️  Duration: {job_duration:.2f} seconds")
        logger.error(f"   ❌ Error: {str(e)}")
        logger.error(f"   🕐 Started: {job_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logger.error(f"   🕐 Failed: {job_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        logger.error("=" * 80)
        app_state.jobs[job_id]["status"] = "FAILED"
        app_state.jobs[job_id]["error"] = str(e)
        app_state.jobs[job_id]["updated_at"] = datetime.utcnow()

@router.post("/translate_file_convo")
async def translate_file_convo(
    files: List[UploadFile] = File(None),
    prompt: str = Form(...),
    target_language: str = Form(None),
    background_tasks: BackgroundTasks = None,
    enable_evaluation: bool = Form(True),
    app_state: AppState = Depends(get_app_state)
):
    """
    Intelligent conversational endpoint with comprehensive intent classification.
    
    ARCHITECTURE:
    User Input → Llama classifies intent → Route to appropriate handler:
      ├─ Chat Intent → /chat endpoint (Llama conversation)
      ├─ Text Translation → /translate_text endpoint (M2M100)
      └─ File Translation → Direct M2M100 translation (current endpoint)
    
    Examples:
      1. "Hello, how are you?" → Chat intent → Llama conversation
      2. 'Translate "Non-disclosure agreement" in Dutch' → Text translation → M2M100
      3. "Translate these files in German and French" → File translation → M2M100
    
    Args:
        target_language: Optional explicit language code. If provided, bypasses ALL Llama processing.
    """
    request_start_time = datetime.utcnow()
    logger.info("=" * 80)
    logger.info(f"🚀 NEW REQUEST RECEIVED at {request_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logger.info(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    logger.info(f"   Files count: {len(files) if files else 0}")
    if files:
        for idx, file in enumerate(files):
            logger.info(f"   File {idx+1}: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'} bytes)")
    logger.info(f"   Target language: {target_language}")
    logger.info(f"   Enable evaluation: {enable_evaluation}")
    logger.info("=" * 80)
    
    if not prompt:
        raise HTTPException(400, "Prompt is required")
    
    # Files are optional - chat and text translation don't require files
    
    # ============================================================================
    # BYPASS: If explicit target_language provided, skip ALL Llama processing
    # ============================================================================
    if target_language and target_language.lower() in LANG_OPTIONS:
        target_lang_key = target_language.lower()
        target_lang_code = LANG_OPTIONS[target_lang_key]
        logger.info(f"✅ [PARALLEL MODE] Explicit language bypass: {target_lang_key} -> {target_lang_code}")
        
        # Validate files are provided for file translation
        if not files:
            raise HTTPException(400, "Files are required when target_language is explicitly provided")
        
        # Jump directly to file translation processing below (skip to line ~1180)
    
    # ============================================================================
    # LLAMA INTENT CLASSIFICATION: Determine what user wants to do
    # ============================================================================
    else:
        intent_prompt = f"""You are an intent classifier for a translation system. Analyze the user prompt and return a JSON object.

**UPLOADED FILES: {len(files) if files else 0}**

User Prompt: "{prompt}"

**STRICT CLASSIFICATION RULES:**

1. **chat** - Choose this if (PRIORITY CHECK FIRST):
   - Questions ABOUT the system/tools: "how to use", "how can I", "what is", "tell me about", "explain"
   - Greetings and conversation: "hello", "hi", "how are you"
   - Help requests: "help me", "guide me", "show me how"
   - Questions about features: "how does translation work", "what can Matterhorn do"
   - **KEY: If asking ABOUT or HOW TO use features → chat (not translation)**
   - Works in ANY language: "翻译工具怎么用?" (asking HOW to use tool) → chat

2. **translate_file** - Choose this if:
   - User uploaded file(s) (count > 0) AND asks to translate them
   - Phrases: "translate this/these file(s)", "translate document(s)", "convert file(s) to [language]"
   - Direct commands: "translate to german", "I need this in french"
   - **PRIORITY: Files uploaded + translation command → translate_file**

3. **translate_text** - Choose this if:
   - NO files uploaded (count = 0) AND
   - User provides quoted text AND asks to translate it
   - Examples: 'translate "hello"', 'what is "bonjour" in english'
   - Specific text to translate: "translate Non-disclosure agreement to dutch"
   - **KEY: Actual text content TO translate → translate_text**

**TARGET LANGUAGES:**

Extract ALL mentioned languages (names OR codes). Valid options:
- Names: {', '.join(LANG_OPTIONS.keys())}
- Codes: zh, de, fr, es, it, en, ja, ko, sv, da, fi, nl
- Handle variations: "chinese", "mandarin", "simplified chinese" → all map to "zh"
- Multiple languages: parse "german, japanese, and italian" OR "de, ja, it"

**IMPORTANT:**
- If files > 0 and user says "translate" → goal MUST be "translate_file"
- Don't overthink - follow the rules strictly
- Extract languages even if mixed with codes (zh, german, etc.)

**OUTPUT FORMAT (JSON only, no markdown):**
{{
  "goal": "translate_file" | "translate_text" | "chat",
  "target_languages": ["german", "japanese", "zh"],
  "text": null or "extracted quoted text",
  "summary": "Brief description of intent"
}}

RESPOND WITH JSON ONLY:
"""
        
        logger.info("🔍 Calling Llama for intent classification...")
        intent_response = await get_llama_response_async(intent_prompt)
        
        if not intent_response:
            logger.warning("❌ Llama classification failed, defaulting to translate_file with english")
            target_lang_key = "english"
            target_lang_code = LANG_OPTIONS[target_lang_key]
        else:
            try:
                import json
                import re
                
                # Clean response (remove markdown code blocks)
                clean_response = intent_response.strip()
                if clean_response.startswith("```"):
                    clean_response = re.sub(r'```(?:json)?\n?', '', clean_response)
                    clean_response = clean_response.strip()
                
                intent_data = json.loads(clean_response)
                logger.info(f"✅ Intent classification: {intent_data}")
                
                goal = intent_data.get("goal", "translate_file")
                target_languages = intent_data.get("target_languages", [])
                extracted_text = intent_data.get("text", None)
                
                # ====================================================================
                # ROUTE 1: CHAT INTENT - User wants to chat with Llama
                # ====================================================================
                if goal == "chat":
                    logger.info("🗨️ Chat intent detected, routing to chat logic")
                    
                    # Detect source language from prompt
                    try:
                        detected_code = detect(prompt[:500])
                        logger.info(f"Detected language code: {detected_code}")
                        
                        # Map detected code to full language name
                        # langdetect returns codes like: en, fr, de, es, it, ja, ko, zh-cn, zh-tw, sv, da, fi, nl
                        code_mapping = {
                            'en': 'English',
                            'fr': 'French', 
                            'de': 'German',
                            'es': 'Spanish',
                            'it': 'Italian',
                            'ja': 'Japanese',
                            'ko': 'Korean',
                            'zh-cn': 'Chinese (Mandarin)',
                            'zh-tw': 'Chinese (Mandarin)',
                            'zh': 'Chinese (Mandarin)',
                            'sv': 'Swedish',
                            'da': 'Danish',
                            'fi': 'Finnish',
                            'nl': 'Dutch'
                        }
                        
                        source_lang_full = code_mapping.get(detected_code, 'English')
                        source_lang_code = detected_code
                        logger.info(f"Language mapped to: {source_lang_full}")
                        
                    except Exception as e:
                        logger.warning(f"Language detection failed: {e}, defaulting to English")
                        source_lang_full = "English"
                        source_lang_code = "en"
                    
                    # Build comprehensive system knowledge base
                    system_knowledge = """
=== MATTERHORN BACK OFFICE SOLUTIONS - SYSTEM KNOWLEDGE BASE ===

You are a helpful AI assistant for Matterhorn Back Office Solutions. You help users with two main tools:

**1. TRANSLATION TOOL** 📝
   Use Cases:
   - Single file, multiple languages: Upload 1 file and translate to multiple languages (e.g., "Translate to German, French, Spanish")
   - Multiple files, single language: Upload multiple files and select one target language
   - Plain text translation: Type or paste text and select target language
   - Conversational translation: Chat with AI and get translations
   
   Supported File Types:
   - DOCX (Microsoft Word)
   - PDF (Portable Document Format)
   - PPTX (PowerPoint Presentations)
   - TXT (Plain Text)
   
   Supported Languages (13 total):
   - English, Chinese (Mandarin/Simplified), German, French, Spanish
   - Italian, Japanese, Korean, Swedish, Danish, Finnish, Dutch
   
   Features:
   - Preserves exact formatting (fonts, colors, tables, headers/footers)
   - AI-powered quality evaluation (6 dimensions: semantic similarity, completeness, fluency, terminology, cultural appropriateness, technical accuracy)
   - Real-time job tracking and status updates
   - Download all completed translations at once
   - Preview translations before downloading
   
   How to Use:
   1. Click on "Translation" button at the bottom
   2. Choose your translation mode:
      - For text: Type your text in the input box, select language from dropdown OR specify in prompt
      - For files: Click the file upload area or drag & drop files
   3. For multi-language: Upload 1 file and type "Translate to German, French, Spanish"
   4. For single language: Upload file(s) and select language from dropdown
   5. Click "Translate" button
   6. Track progress in "Translation Jobs" section
   7. Download completed files or preview them

**2. TIMESHEET TOOL** ⏱️
   Use Cases:
   - Create timesheet entries through conversational AI
   - Track billable hours for clients and matters
   - Manage expenses and activities
   - View and edit existing entries
   
   Features:
   - Conversational entry creation (AI asks questions one by one)
   - Client and matter tracking
   - Phase/Task categorization
   - Activity logging
   - Expense tracking
   - Duplicate entries for repeated tasks
   - Filter and search entries
   
   How to Use:
   1. Click on "Timesheet" button at the bottom
   2. AI will guide you through creating an entry by asking:
      - Client name
      - Matter/Project
      - Date
      - Hours worked/billed
      - Description
      - Rate and currency
   3. Confirm the entry when prompted
   4. View all entries by clicking "Entries" button

**3. OTHER FEATURES**
   - Matters: Search and manage legal matters
   - Entries: View timesheet entries
   - Multi-language support throughout the platform

When users ask about these tools, provide helpful, step-by-step guidance in a friendly conversational manner. Ask follow-up questions to understand their needs better and guide them to the right feature.
"""
                    
                    # Build chat prompt with system knowledge and STRICT language instruction
                    chat_prompt = f"""You are a helpful AI assistant for Matterhorn Back Office Solutions.

**CRITICAL LANGUAGE INSTRUCTION:**
The user wrote their message in {source_lang_full}. You MUST respond in {source_lang_full} ONLY.
DO NOT translate the user's message. DO NOT respond in English unless the user wrote in English.
If user writes in French → respond in French
If user writes in German → respond in German  
If user writes in Chinese → respond in Chinese
And so on for all languages.

**SYSTEM KNOWLEDGE:**
{system_knowledge}

**YOUR TASK:**
1. Read the user's message below (it's in {source_lang_full})
2. Understand what they need help with
3. Provide helpful, conversational guidance in {source_lang_full}
4. Be friendly and ask follow-up questions
5. Give step-by-step instructions when explaining features

**USER MESSAGE ({source_lang_full}):**
{prompt}

**YOUR RESPONSE (in {source_lang_full} language):**
"""
                    
                    # Get Llama response
                    chat_response = await get_llama_response_async(chat_prompt)
                    
                    if not chat_response:
                        raise HTTPException(503, "Chat service unavailable (Ollama error)")
                    
                    # Update mock database
                    mock_db_update()
                    
                    return JSONResponse({
                        "response": chat_response,
                        "intent": "chat",
                        "source_language": source_lang_full,
                        "jobs": []
                    })
                
                # ====================================================================
                # ROUTE 2: TEXT TRANSLATION - Translate quoted/extracted text
                # ====================================================================
                elif goal == "translate_text":
                    logger.info("📝 Text translation intent detected")
                    
                    if not extracted_text or not extracted_text.strip():
                        raise HTTPException(400, "No text found to translate. Please provide text in quotes.")
                    
                    # Get target language
                    if target_languages and len(target_languages) > 0:
                        first_lang = target_languages[0].lower().strip()
                        target_lang_key = first_lang if first_lang in LANG_OPTIONS else "english"
                    else:
                        raise HTTPException(400, "No target language specified for text translation")
                    
                    target_lang_code = LANG_OPTIONS[target_lang_key]
                    
                    # Auto-detect source language
                    try:
                        source_lang_code = detect(extracted_text[:500])
                        if source_lang_code not in LANG_OPTIONS.values():
                            source_lang_code = "en"
                    except:
                        source_lang_code = "en"
                    
                    logger.info(f"📝 Translating text from {source_lang_code} to {target_lang_code}")
                    
                    # Check if source and target are the same
                    if source_lang_code == target_lang_code:
                        return JSONResponse({
                            "translated_text": extracted_text,
                            "source_language": source_lang_code,
                            "target_language": target_lang_key,
                            "intent": "translate_text",
                            "message": "Source and target languages are the same. No translation needed."
                        })
                    
                    # Translate using M2M100
                    loop = asyncio.get_event_loop()
                    translated_text = await loop.run_in_executor(
                        app_state.translation_executor,
                        partial(translate_with_rules, extracted_text, target_lang_code, MAX_RETRIES, app_state, app_state.kb_en_to_zh)
                    )
                    
                    return JSONResponse({
                        "translated_text": translated_text,
                        "source_language": source_lang_code,
                        "target_language": target_lang_key,
                        "intent": "translate_text",
                        "source_text_length": len(extracted_text),
                        "translated_text_length": len(translated_text),
                        "message": "Text translation complete."
                    })
                
                # ====================================================================
                # ROUTE 3: FILE TRANSLATION - Translate uploaded files
                # ====================================================================
                elif goal == "translate_file":
                    logger.info("📄 File translation intent detected")
                    
                    if not files:
                        raise HTTPException(400, "No files uploaded for file translation")
                    
                    # Extract first language (frontend handles multi-language by calling multiple times)
                    if target_languages and len(target_languages) > 0:
                        first_lang = target_languages[0].lower().strip()
                        
                        # Direct lookup or fuzzy match
                        if first_lang in LANG_OPTIONS:
                            target_lang_key = first_lang
                        else:
                            # Fuzzy match for variations
                            target_lang_key = next(
                                (k for k in LANG_OPTIONS.keys() if k.startswith(first_lang) or first_lang in k),
                                "english"
                            )
                        
                        target_lang_code = LANG_OPTIONS[target_lang_key]
                        logger.info(f"✅ Using detected language: {first_lang} -> {target_lang_key} -> {target_lang_code}")
                    else:
                        logger.warning("No languages detected, defaulting to English")
                        target_lang_key = "english"
                        target_lang_code = LANG_OPTIONS[target_lang_key]
                    
                    # Continue to file translation processing below
                else:
                    # Unknown goal - default to file translation if files present, else error
                    logger.warning(f"Unknown goal '{goal}', attempting to infer from context")
                    if files:
                        logger.info("Files present, defaulting to file translation")
                        target_lang_key = "english"
                        target_lang_code = LANG_OPTIONS[target_lang_key]
                    else:
                        raise HTTPException(400, f"Cannot process intent '{goal}' without files")
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"❌ Intent parsing failed: {e}")
                logger.error(f"Raw response: {intent_response[:500]}")
                
                # Fallback: If files present, assume file translation
                if files:
                    logger.info("Fallback: Files present, defaulting to file translation")
                    target_lang_key = "english"
                    target_lang_code = LANG_OPTIONS[target_lang_key]
                else:
                    raise HTTPException(400, f"Intent classification failed and no files uploaded: {str(e)}")
    
    # ============================================================================
    # OLD FLOW (COMMENTED OUT - Kept for reference)
    # ============================================================================
    # else:
    #     # OLD: Single language detection with Llama
    #     intent_prompt = f"""Analyze the following prompt and determine the target language for translation (must be one of: {', '.join(LANG_OPTIONS.keys())}). If no language is specified, default to 'english'. Respond with only the target language name in lowercase.
    #
    #Prompt: {prompt}"""
    #     
    #     intent_response = await get_llama_response_async(intent_prompt)
    #     target_lang_key = intent_response.strip().lower() if intent_response else "english"
    #     
    #     if target_lang_key not in LANG_OPTIONS:
    #         logger.warning(f"Invalid or undetected target language '{target_lang_key}', defaulting to English")
    #         target_lang_key = "english"
    #     
    #     target_lang_code = LANG_OPTIONS[target_lang_key]
    #     logger.info(f"LLM detected target language: {target_lang_key} -> {target_lang_code}")
    
    # Queue jobs for background processing
    job_entries = []
    logger.info(f"📋 [PARALLEL MODE] Processing {len(files)} file(s) for translation to {target_lang_code}")
    
    for file_idx, file in enumerate(files, 1):
        ext = os.path.splitext(file.filename)[1].lower()
        logger.info(f"   📄 Processing file {file_idx}/{len(files)}: {file.filename} (type: {ext})")
        
        if ext not in [".docx", ".pdf", ".pptx", ".txt"]:
            logger.warning(f"   ❌ Unsupported file type: {ext} for {file.filename}")
            job_entries.append({
                "filename": file.filename,
                "status": "FAILED",
                "error": f"Unsupported file type: {ext}"
            })
            continue
        
        # Save file to temp location
        logger.info(f"   💾 Saving file to temporary location...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"   ✅ File saved to: {tmp_path}")
        
        # Create job entry
        job_id = str(uuid.uuid4())
        job_created_at = datetime.utcnow()
        app_state.jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "target_language": target_lang_code,
            "status": "QUEUED",
            "created_at": job_created_at,
            "updated_at": job_created_at,
        }
        
        logger.info(f"   🆔 Job created: {job_id}")
        logger.info(f"   📝 Job details: filename={file.filename}, target_lang={target_lang_code}, status=QUEUED")
        
        # Add to background tasks
        logger.info(f"   ⚡ Queuing background task for job {job_id}...")
        background_tasks.add_task(
            run_translation_job_background,
            job_id,
            file.filename,
            tmp_path,
            target_lang_code,
            enable_evaluation,
            app_state
        )
        logger.info(f"   ✅ Background task queued for job {job_id}")
        
        job_entries.append({
            "filename": file.filename,
            "target_language": target_lang_code,
            "job_id": job_id
        })
    
    # Return immediately with job tracking info
    num_files = len(job_entries)
    request_end_time = datetime.utcnow()
    request_duration = (request_end_time - request_start_time).total_seconds()
    
    logger.info("=" * 80)
    logger.info(f"✅ REQUEST COMPLETED in {request_duration:.3f} seconds")
    logger.info(f"   Created {num_files} translation job(s) for target language: {target_lang_code}")
    logger.info(f"   Jobs: {[job['job_id'] for job in job_entries]}")
    logger.info(f"   ⚡ All jobs are now running in PARALLEL via ThreadPoolExecutor")
    logger.info("=" * 80)
    
    return JSONResponse({
        "response": f"{num_files} translation job(s) started for files to {target_lang_code}. Track progress via /status/{{job_id}}.",
        "jobs": job_entries
    })