# from typing import Dict, Any, Optional
# from concurrent.futures import ThreadPoolExecutor
# import torch
# from fastapi import Request  # Use Request instead of FastAPI

# class AppState:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#         self.device = None
#         self.kb_en_to_zh: Optional[Dict[str, str]] = None
#         self.nltk_available = False
#         self.translation_executor = ThreadPoolExecutor(max_workers=2)
#         self.translated_files: Dict[str, Dict[str, Any]] = {}
#         self.jobs: Dict[str, Dict[str, Any]] = {}
#         self.evaluation_results: Dict[str, Dict[str, Any]] = {}

# def get_app_state(request: Request) -> AppState:
#     """
#     Dependency to retrieve AppState from FastAPI app state via Request.
#     """
#     return request.app.state.app_state


# state.py
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import torch
from fastapi import Request
import multiprocessing
import logging

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.kb_en_to_zh: Optional[Dict[str, str]] = None
        self.nltk_available = False
        
        # Dynamic worker count based on CPU cores
        # Use 4x CPU cores for I/O-bound translation tasks, max 16 (tune as needed)
        try:
            cpu_count = multiprocessing.cpu_count() or 1
        except Exception:
            cpu_count = 1
        translation_workers = min(cpu_count * 4, 16)
        file_io_workers = min(8, max(2, cpu_count))

        logger.info(f"Initializing executors: translation_workers={translation_workers}, file_io_workers={file_io_workers}")

        # Executor used for translation / model-bound tasks
        self.translation_executor = ThreadPoolExecutor(
            max_workers=translation_workers,
            thread_name_prefix="translation_worker"
        )
        
        # Executor used for file I/O, conversions, and disk operations
        self.file_io_executor = ThreadPoolExecutor(
            max_workers=file_io_workers,
            thread_name_prefix="file_io_worker"
        )
        
        # State containers (unchanged)
        self.translated_files: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        
        # Optional: scheduling helpers (can be used later)
        self.job_queue = []
        self.active_jobs = set()

def get_app_state(request: Request) -> AppState:
    """Dependency to retrieve AppState from FastAPI app state via Request."""
    return request.app.state.app_state
