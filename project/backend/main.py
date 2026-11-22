from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
import os
from datetime import datetime
from .routes import router
from .state import AppState, get_app_state
from .utils import setup_nltk, load_combined_kb
from .translation import load_translation_model
from .config import evaluation_config


# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import fitz  # PyMuPDF
# Backward-compatibility patch for pdf2docx expecting Rect.get_area()
if not hasattr(fitz.Rect, "get_area"):
    fitz.Rect.get_area = lambda self: self.width * self.height


app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",        # Frontend (local)
        "http://127.0.0.1:5173",
        "http://13.203.104.235:5173",   # Frontend (production IPs)
        "http://13.203.251.172:5173",
        "http://localhost:8002",        # MatterAI Backend (local)
        "http://127.0.0.1:8002",
        "http://13.203.104.235:8002",   # MatterAI Backend (production IPs)
        "http://13.203.251.172:8002"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize AppState
app.state.app_state = AppState()

async def cleanup_jobs(app_state: AppState, expiry_minutes: int = 30):
    """
    Periodically clean up old completed/failed jobs and their associated files.
    Runs every 5 minutes and removes jobs older than expiry_minutes.
    """
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            now = datetime.utcnow()
            to_delete = []
            
            for job_id, job in list(app_state.jobs.items()):
                updated_at = job.get("updated_at")
                if not updated_at:
                    continue
                
                # Calculate age in minutes
                age = (now - updated_at).total_seconds() / 60
                
                if job["status"] in ["COMPLETED", "FAILED"] and age > expiry_minutes:
                    # Clean up associated file
                    if job.get("path") and os.path.exists(job["path"]):
                        try:
                            os.remove(job["path"])
                            logger.info(f"Cleaned up job file: {job['path']}")
                        except Exception as e:
                            logger.warning(f"Failed to delete job file {job['path']}: {e}")
                    
                    to_delete.append(job_id)
            
            # Remove expired jobs
            for job_id in to_delete:
                del app_state.jobs[job_id]
                logger.info(f"Removed expired job: {job_id}")
            
            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} expired job(s)")
                
        except Exception as e:
            logger.error(f"Error in cleanup_jobs: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying if there's an error

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application and loading resources...")
    
    app_state = app.state.app_state
    app_state.kb_en_to_zh = load_combined_kb([
        "corrected_mandarin/corrected_mandarin.json",
        "corrected_mandarin/Traditional_mandarin.json"
    ])
    app_state.nltk_available = setup_nltk()
    load_translation_model(app_state)
    
    logger.info(f"Translation executor initialized with {app_state.translation_executor._max_workers} workers")
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    logger.info("Shutting down translation executor...")
    # app_state.translation_executor.shutdown(wait=True)
    logger.info("Shutting down translation_executor...")
    try:
        app_state.translation_executor.shutdown(wait=True)
    except Exception as e:
        logger.warning(f"Error shutting down translation_executor: {e}")

    logger.info("Shutting down file_io_executor...")
    try:
        app_state.file_io_executor.shutdown(wait=True)
    except Exception as e:
        logger.warning(f"Error shutting down file_io_executor: {e}")

    logger.info("Translation executor shutdown complete")
    
    logger.info("Cleaning up temporary files...")
    for data in app_state.translated_files.values():
        try:
            if os.path.exists(data["path"]):
                os.unlink(data["path"])
        except Exception as e:
            logger.warning(f"Failed to cleanup file {data.get('path')}: {e}")
    
    logger.info("Application shutdown complete")

app.router.lifespan_context = lifespan

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_jobs(app.state.app_state))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)