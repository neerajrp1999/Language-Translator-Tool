import uuid
from datetime import datetime
import logging
from typing import Dict, Any
from .config import evaluation_config
from .ollama import get_llama_response_async
from .state import AppState
from .utils import clean_json_response
import json
import re

logger = logging.getLogger(__name__)

async def evaluate_translation_with_llama(
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
    source_filename: str,
    translated_filename: str
) -> Dict[str, Any]:
    try:
        # Prepare comprehensive evaluation prompt using configuration
        evaluation_prompt = f"""{evaluation_config.prompts['evaluation_instructions'].format(source_lang=source_lang, target_lang=target_lang)}

SOURCE TEXT ({source_lang}):
{source_text[:evaluation_config.text_limits['source_text_sample']]}

TRANSLATED TEXT ({target_lang}):
{translated_text[:evaluation_config.text_limits['translated_text_sample']]}

Evaluate the translation quality across these 6 dimensions (score 0-100 for each):

1. **SEMANTIC SIMILARITY** ({evaluation_config.weights['semantic_similarity']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['semantic_similarity']}

2. **COMPLETENESS** ({evaluation_config.weights['completeness']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['completeness']}

3. **FLUENCY** ({evaluation_config.weights['fluency']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['fluency'].format(target_lang=target_lang)}

4. **TERMINOLOGY** ({evaluation_config.weights['terminology']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['terminology']}

5. **CULTURAL APPROPRIATENESS** ({evaluation_config.weights['cultural_appropriateness']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['cultural_appropriateness']}

6. **TECHNICAL ACCURACY** ({evaluation_config.weights['technical_accuracy']:.0%} weight): {evaluation_config.prompts['dimension_instructions']['technical_accuracy']}

Respond ONLY with this JSON structure (NO COMMENTS, PURE JSON ONLY):
```json
{{
  "semantic_similarity": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "missing_concepts": ["specific concept 1 that was lost", "specific concept 2 that was lost"],
    "added_concepts": ["concept that was added but not in original"],
    "semantic_accuracy": "excellent|good|fair|poor"
  }},
  "completeness": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "missing_sections": ["specific section/paragraph that was omitted", "specific sentence that was skipped"],
    "added_sections": ["content that was added but not in original"],
    "structural_issues": ["specific structural problem 1", "specific structural problem 2"],
    "completeness_level": "complete|mostly_complete|partial|incomplete"
  }},
  "fluency": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "grammar_issues": ["specific grammar error 1", "specific grammar error 2"],
    "awkward_phrases": ["specific awkward phrase 1", "specific awkward phrase 2"],
    "fluency_rating": "excellent|good|fair|poor"
  }},
  "terminology": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "incorrect_terms": ["term1: incorrect_translation -> correct_translation", "term2: wrong -> right"],
    "inconsistent_terms": ["term that was translated differently in different places"],
    "terminology_rating": "excellent|good|fair|poor"
  }},
  "cultural_appropriateness": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "cultural_issues": ["specific cultural mismatch 1", "specific inappropriate expression 2"],
    "localization_quality": "excellent|good|fair|poor"
  }},
  "technical_accuracy": {{
    "score": <0-100>,
    "analysis": "<detailed analysis with specific examples>",
    "technical_errors": ["specific number error", "specific date format issue", "specific unit mistake"],
    "accuracy_level": "perfect|good|fair|poor"
  }},
  "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
  "improvement_areas": ["specific area 1", "specific area 2", "specific area 3"],
  "overall_assessment": "<comprehensive summary with specific examples>"
}}
```"""

        # Call Ollama API with PURE evaluation configuration
        llama_response = await get_llama_response_async(evaluation_prompt, use_evaluation_config=True)
        
        if not llama_response:
            logger.error("Llama evaluation failed - no response")
            return generate_fallback_evaluation(source_filename, translated_filename, source_lang, target_lang, "No LLM response")
        
        # Parse JSON response with robust error handling
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*\n([\s\S]*?)\n```', llama_response, re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON object boundaries
                json_obj_match = re.search(r'\{[\s\S]*\}', llama_response)
                if json_obj_match:
                    json_str = json_obj_match.group(0)
                else:
                    json_str = llama_response.strip()
            
            # Clean JSON
            json_str = clean_json_response(json_str)
            
            # Parse the cleaned JSON
            evaluation_data = json.loads(json_str)
            
            # Validate and fill missing keys
            required_keys = ['semantic_similarity', 'completeness', 'fluency', 'terminology', 'cultural_appropriateness', 'technical_accuracy']
            missing_keys = [key for key in required_keys if key not in evaluation_data]
            for key in missing_keys:
                evaluation_data[key] = {
                    'score': evaluation_config.fallback_scores['default_score'],
                    'analysis': 'Not evaluated',
                    'rating': 'unknown'
                }
            
            # Ensure arrays have content
            array_fields = {
                'semantic_similarity': ['missing_concepts', 'added_concepts'],
                'completeness': ['missing_sections', 'added_sections', 'structural_issues'],
                'fluency': ['grammar_issues', 'awkward_phrases'],
                'terminology': ['incorrect_terms', 'inconsistent_terms'],
                'cultural_appropriateness': ['cultural_issues'],
                'technical_accuracy': ['technical_errors']
            }
            
            for metric, fields in array_fields.items():
                if metric in evaluation_data:
                    for field in fields:
                        if field in evaluation_data[metric]:
                            if not evaluation_data[metric][field] or len(evaluation_data[metric][field]) == 0:
                                evaluation_data[metric][field] = [evaluation_config.messages['no_issues_detected']]
            
            # Ensure top-level arrays have content
            if 'strengths' in evaluation_data and (not evaluation_data['strengths'] or len(evaluation_data['strengths']) == 0):
                evaluation_data['strengths'] = [evaluation_config.messages['translation_completed']]
            
            if 'improvement_areas' in evaluation_data and (not evaluation_data['improvement_areas'] or len(evaluation_data['improvement_areas']) == 0):
                evaluation_data['improvement_areas'] = [evaluation_config.messages['no_improvements']]
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed after cleaning: {e}")
            logger.debug(f"Raw response (first {evaluation_config.text_limits['debug_sample']} chars): {llama_response[:evaluation_config.text_limits['debug_sample']]}")
            return generate_fallback_evaluation(source_filename, translated_filename, source_lang, target_lang, f"JSON parse error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}")
            return generate_fallback_evaluation(source_filename, translated_filename, source_lang, target_lang, f"Parsing error: {str(e)}")
        
        # Calculate weighted combined accuracy
        combined_accuracy = sum(
            evaluation_data.get(metric, {}).get('score', evaluation_config.fallback_scores['default_score']) * weight
            for metric, weight in evaluation_config.weights.items()
        )
        
        # Generate evaluation ID
        evaluation_id = str(uuid.uuid4())
        
        # Construct comprehensive evaluation result
        evaluation_result = {
            'evaluation_id': evaluation_id,
            'source_filename': source_filename,
            'translated_filename': translated_filename,
            'source_language': source_lang,
            'target_language': target_lang,
            'timestamp': datetime.utcnow().isoformat(),
            'combined_accuracy': round(combined_accuracy, 2),
            'individual_scores': {
                'semantic_similarity': evaluation_data.get('semantic_similarity', {}),
                'completeness': evaluation_data.get('completeness', {}),
                'fluency': evaluation_data.get('fluency', {}),
                'terminology': evaluation_data.get('terminology', {}),
                'cultural_appropriateness': evaluation_data.get('cultural_appropriateness', {}),
                'technical_accuracy': evaluation_data.get('technical_accuracy', {})
            },
            'detailed_analysis': {
                'strengths': evaluation_data.get('strengths', []),
                'improvement_areas': evaluation_data.get('improvement_areas', []),
                'overall_assessment': evaluation_data.get('overall_assessment', 'Evaluation completed')
            },
            'evaluation_method': f'Pure Ollama Evaluation ({evaluation_config.model_config["model_name"]})',
            'source_text_sample': source_text[:evaluation_config.text_limits['preview_sample']] + "..." if len(source_text) > evaluation_config.text_limits['preview_sample'] else source_text,
            'translated_text_sample': translated_text[:evaluation_config.text_limits['preview_sample']] + "..." if len(translated_text) > evaluation_config.text_limits['preview_sample'] else translated_text
        }
        
        logger.info(f"Evaluation complete: {evaluation_id}, Combined accuracy: {combined_accuracy:.2f}%")
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Llama evaluation failed: {str(e)}")
        return generate_fallback_evaluation(source_filename, translated_filename, source_lang, target_lang, str(e))

def generate_fallback_evaluation(source_filename: str, translated_filename: str, source_lang: str, target_lang: str, error_msg: str) -> Dict[str, Any]:
    evaluation_id = str(uuid.uuid4())
    default_score = evaluation_config.fallback_scores['default_score']
    fallback_accuracy = evaluation_config.fallback_scores['fallback_combined_accuracy']
    
    return {
        'evaluation_id': evaluation_id,
        'source_filename': source_filename,
        'translated_filename': translated_filename,
        'source_language': source_lang,
        'target_language': target_lang,
        'timestamp': datetime.utcnow().isoformat(),
        'combined_accuracy': fallback_accuracy,
        'individual_scores': {
            'semantic_similarity': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'semantic_accuracy': 'unknown'},
            'completeness': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'completeness_level': 'unknown'},
            'fluency': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'fluency_rating': 'unknown'},
            'terminology': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'terminology_rating': 'unknown'},
            'cultural_appropriateness': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'localization_quality': 'unknown'},
            'technical_accuracy': {'score': default_score, 'analysis': evaluation_config.messages['evaluation_unavailable'], 'accuracy_level': 'unknown'}
        },
        'detailed_analysis': {
            'strengths': [],
            'improvement_areas': ['Evaluation system unavailable'],
            'overall_assessment': f'Evaluation failed: {error_msg}. Ensure Ollama is running with {evaluation_config.model_config["model_name"]} model.'
        },
        'evaluation_method': evaluation_config.messages['fallback_method'],
        'error': error_msg
    }

# DEPRECATED functions (kept as per original, but commented)
# def calculate_word_level_accuracy(original: str, back_translated: str) -> Dict[str, Any]:
#     pass
# ... (all other deprecated functions similarly commented)