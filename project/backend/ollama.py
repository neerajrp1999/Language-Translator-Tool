import requests
import httpx
import logging
import asyncio
import re
import json
from typing import Optional
from .config import evaluation_config

logger = logging.getLogger(__name__)

def get_llama_response(prompt: str, model: str = "llama3.1:8b", expect_json: bool = False) -> Optional[str]:
    """Query Ollama with retries and extract JSON from response (synchronous version)."""
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=data, timeout=30)
            if resp.status_code == 200:
                raw_response = resp.json()["response"].strip()
                # Extract JSON block using regex
                json_match = re.search(r'```json\n([\s\S]*?)\n```', raw_response)
                if json_match:
                    json_str = json_match.group(1).strip()
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        logger.error(f"Extracted JSON is invalid: {json_str}")
                else:
                    # Only warn if JSON was expected
                    if expect_json:
                        logger.warning(f"No JSON block found in response: {raw_response[:100]}...")
                    else:
                        logger.debug(f"Response received (plain text): {raw_response[:100]}...")
                return raw_response
            else:
                logger.error(f"Ollama error (attempt {attempt + 1}): {resp.text}")
        except Exception as e:
            logger.error(f"Ollama request failed (attempt {attempt + 1}): {str(e)}")
            if attempt == 2:
                return None
    return None

async def get_llama_response_async(prompt: str, model: str = "llama3.1:8b", use_evaluation_config: bool = False) -> Optional[str]:
    url = "http://localhost:11434/api/generate"
    
    # Evaluation config means we expect JSON
    expect_json = use_evaluation_config
    
    if use_evaluation_config:
        data = {
            "model": evaluation_config.model_config['model_name'],
            "prompt": prompt,
            "stream": False,
            "options": evaluation_config.get_ollama_options()
        }
        timeout = evaluation_config.model_config['timeout']
    else:
        data = {"model": model, "prompt": prompt, "stream": False}
        timeout = 30.0  # Default
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, json=data)
                if resp.status_code == 200:
                    raw_response = resp.json()["response"].strip()
                    json_match = re.search(r'```json\n([\s\S]*?)\n```', raw_response)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            logger.error(f"Extracted JSON is invalid: {json_str}")
                    else:
                        # Only warn if JSON was expected (evaluation mode)
                        if expect_json:
                            logger.warning(f"No JSON block found in response: {raw_response[:100]}...")
                        else:
                            logger.debug(f"Response received (plain text): {raw_response[:100]}...")
                    return raw_response
                else:
                    logger.error(f"Ollama error (attempt {attempt + 1}): {resp.text}")
            except Exception as e:
                logger.error(f"Ollama request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == 2:
                    return None
                await asyncio.sleep(0.5 * (2 ** attempt))
    return None