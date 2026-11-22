import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Hardcoded settings
LIBREOFFICE_PATH = r"/usr/lib/libreoffice/program/soffice.bin"  # Updated for Ubuntu
USE_XML_METHOD = False
MAX_RETRIES = 2
CHUNK_SIZE = 400
ENABLE_PARALLEL = True  # Enabled for parallel processing
OLLAMA_TIMEOUT = 30.0  # Default, overridden by EvaluationConfig if needed

# Language mapping
LANG_OPTIONS: Dict[str, str] = {
    "chinese (mandarin)": "zh",
    "simplified chinese": "zh",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "dutch": "nl"
}

class EvaluationConfig:
    """Configuration for evaluation system"""
    def __init__(self):
        # Evaluation weights (must sum to 1.0)
        self.weights = {
            'semantic_similarity': 0.30,
            'completeness': 0.20,
            'fluency': 0.20,
            'terminology': 0.15,
            'cultural_appropriateness': 0.10,
            'technical_accuracy': 0.05
        }
        
        # Quality grade thresholds (0-100)
        self.quality_thresholds = {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'needs_improvement': 0
        }
        
        # Text processing limits
        self.text_limits = {
            'source_text_sample': 1500,
            'translated_text_sample': 1500,
            'language_detection_sample': 500,
            'preview_sample': 300,
            'debug_sample': 500
        }
        
        # Fallback scores
        self.fallback_scores = {
            'default_score': 50,
            'fallback_combined_accuracy': 50.0
        }
        
        # Ollama model configuration - ALL evaluation parameters driven by Ollama
        self.model_config = {
            'model_name': 'llama3.1:8b',
            'timeout': 30.0,
            'temperature': 0.1,  # Low temperature for consistent evaluation
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'stop': ['```', '\n\n'],  # Stop tokens for JSON responses
            'num_ctx': 4096,  # Context window
            'num_predict': 2048,  # Max tokens to generate
            'seed': 42  # For reproducible evaluations
        }
        
        # Evaluation prompts
        self.prompts = {
            'evaluation_instructions': """You are an expert translation quality evaluator. Analyze this translation from {source_lang} to {target_lang} with EXTREME DETAIL.

IMPORTANT: Provide SPECIFIC EXAMPLES for ALL arrays. Do NOT leave any array empty. If no issues found, write "None found" or "No issues detected".""",
            
            'dimension_instructions': {
                'semantic_similarity': """How well does the translation preserve the original meaning? Consider:
   - Core message preservation
   - Concept accuracy
   - Contextual equivalence
   - Paraphrasing appropriateness
   - Look for: lost concepts, added concepts, meaning shifts""",
                
                'completeness': """Is all content translated without omissions or additions?
   - Missing sections, paragraphs, sentences
   - Added content not in original
   - Structural integrity
   - Information preservation
   - Look for: skipped parts, extra content, structural changes""",
                
                'fluency': """Does the translation read naturally in {target_lang}?
   - Grammar correctness
   - Sentence structure
   - Word choice naturalness
   - Reading flow
   - Look for: grammar errors, awkward constructions, unnatural phrasing""",
                
                'terminology': """Are technical and domain-specific terms correct?
   - Technical term accuracy
   - Consistency across document
   - Domain appropriateness
   - Proper noun handling
   - Look for: wrong technical terms, inconsistent translations, proper noun errors""",
                
                'cultural_appropriateness': """Is the translation culturally adapted?
   - Cultural references
   - Idiomatic expressions
   - Localization quality
   - Cultural sensitivity
   - Look for: cultural mismatches, inappropriate expressions, localization issues""",
                
                'technical_accuracy': """Are numbers, dates, and technical details preserved?
   - Number accuracy
   - Date/time formats
   - Units of measurement
   - Technical specifications
   - Look for: wrong numbers, format errors, unit mistakes"""
            }
        }
        
        # Default messages
        self.messages = {
            'no_issues_detected': "No issues detected in this area",
            'translation_completed': "Translation completed successfully",
            'no_improvements': "No major improvement areas identified",
            'evaluation_unavailable': "Evaluation unavailable",
            'fallback_method': "Fallback (Llama unavailable)"
        }
    
    def validate_weights(self):
        """Validate that weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        return True
    
    def update_model_config(self, new_config: dict):
        """Update model configuration with validation"""
        if 'model_name' in new_config:
            self.model_config['model_name'] = new_config['model_name']
        if 'timeout' in new_config and new_config['timeout'] > 0:
            self.model_config['timeout'] = new_config['timeout']
        if 'temperature' in new_config and 0 <= new_config['temperature'] <= 2:
            self.model_config['temperature'] = new_config['temperature']
        if 'top_p' in new_config and 0 < new_config['top_p'] <= 1:
            self.model_config['top_p'] = new_config['top_p']
        # Add other parameter validations as needed
    
    def get_ollama_options(self):
        """Get Ollama options dictionary for API calls"""
        return {
            "temperature": self.model_config['temperature'],
            "top_p": self.model_config['top_p'],
            "top_k": self.model_config['top_k'],
            "repeat_penalty": self.model_config['repeat_penalty'],
            "stop": self.model_config['stop'],
            "num_ctx": self.model_config['num_ctx'],
            "num_predict": self.model_config['num_predict'],
            "seed": self.model_config['seed']
        }

# Initialize evaluation configuration
evaluation_config = EvaluationConfig()
evaluation_config.validate_weights()  # Validate on init