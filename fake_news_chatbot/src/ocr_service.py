"""
OCR service using PaddleOCR for text extraction from images.
Adapted from newsModel/extractor.py to work with PIL Images.
"""

import os
import re
import tempfile
import logging
from PIL import Image
from paddleocr import PaddleOCRVL

# Bypass connectivity check to speed up initialization
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Reduce PaddleOCR logging to avoid Tensor conversion issues
logging.getLogger('paddle').setLevel(logging.WARNING)
logging.getLogger('ppocr').setLevel(logging.WARNING)
logging.getLogger('paddlex').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global pipeline instance (singleton pattern)
_pipeline = None


def _get_pipeline():
    """
    Get or initialize the PaddleOCR pipeline (singleton pattern).
    
    Returns:
        PaddleOCRVL: Initialized pipeline instance
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing PaddleOCR PaddleOCRVL model...")
        # Suppress logging during initialization
        old_levels = {}
        for logger_name in ['paddle', 'ppocr', 'paddlex']:
            logger_obj = logging.getLogger(logger_name)
            old_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.ERROR)
        
        try:
            _pipeline = PaddleOCRVL()
            logger.info("PaddleOCR model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR model: {e}")
            raise
        finally:
            # Restore logging levels
            for logger_name, old_level in old_levels.items():
                logging.getLogger(logger_name).setLevel(old_level)
    return _pipeline


def format_newspaper_structure(parsing_res_list):
    """
    Format OCR results into a structured newspaper description.
    
    Args:
        parsing_res_list: List of parsed blocks from OCR results
        
    Returns:
        Formatted string describing the newspaper content
    """
    newspaper_name = None
    date = None
    main_headline = None
    subheadlines = []
    section_labels = []
    weather_info = None
    website_info = None
    
    # Extract information from parsing results
    for block in parsing_res_list:
        # Handle both dictionary and PaddleOCRVLBlock object formats
        if isinstance(block, dict):
            label = block.get('block_label', '')
            content = block.get('block_content', '').strip()
        else:
            # Handle PaddleOCRVLBlock object
            # Try to get label and content from object attributes
            label = ''
            content = ''
            
            # Try common attribute names
            if hasattr(block, 'block_label'):
                label = block.block_label
            elif hasattr(block, 'label'):
                label = block.label
            elif hasattr(block, 'type'):
                label = block.type
            
            if hasattr(block, 'block_content'):
                content = block.block_content
            elif hasattr(block, 'content'):
                content = block.content
            elif hasattr(block, 'text'):
                content = block.text
            
            # Convert to string and strip
            if content:
                content = str(content).strip()
        
        if not content:
            continue
            
        # Extract newspaper name (usually paragraph_title at top)
        if label == 'paragraph_title' and not newspaper_name:
            # Check if it looks like a newspaper name
            if any(word in content.upper() for word in ['TIMES', 'POST', 'NEWS', 'JOURNAL', 'TRIBUNE', 'HERALD']):
                newspaper_name = content
        
        # Extract date (usually in text blocks with date-like patterns)
        if not date and label == 'text':
            # Look for date patterns
            date_patterns = [
                r'([A-Z]+DAY,\s+[A-Z]+\s+\d{1,2},\s+\d{4})',  # WEDNESDAY, AUGUST 25, 2010
                r'([A-Z]+\s+\d{1,2},\s+\d{4})',  # AUGUST 25, 2010
                r'(\d{1,2}/\d{1,2}/\d{4})',  # 08/25/2010
            ]
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    date = match.group(1)
                    break
        
        # Extract main headline (doc_title)
        if label == 'doc_title' and not main_headline:
            main_headline = content
        
        # Extract subheadlines (paragraph_title that's not the newspaper name)
        if label == 'paragraph_title' and content != newspaper_name:
            subheadlines.append(content)
        
        # Extract section labels
        if label == 'text' and len(content) < 20 and content.isupper():
            section_labels.append(content)
        
        # Extract weather info
        if 'SHOWER' in content.upper() or 'HIGH' in content.upper() or 'LOW' in content.upper():
            weather_info = content
        
        # Extract website info
        if '.com' in content.lower() or 'www.' in content.lower():
            website_info = content
    
    # Build structured description
    description_parts = []
    
    # Opening description
    description_parts.append("This image appears to be the front page of a newspaper.")
    description_parts.append("")
    
    # Newspaper name
    if newspaper_name:
        description_parts.append(f"Newspaper: {newspaper_name}.")
    else:
        description_parts.append("Newspaper: [Not identified].")
    
    # Date
    if date:
        description_parts.append(f"Date: {date}.")
    else:
        description_parts.append("Date: [Not identified].")
    
    description_parts.append("")
    
    # Main headline
    if main_headline:
        description_parts.append(f"Main Story: {main_headline}")
        description_parts.append("")
        description_parts.append(f"Headline: \"{main_headline}\"")
    
    # Subheadlines
    if subheadlines:
        description_parts.append("")
        description_parts.append("Additional Headlines:")
        for i, subheadline in enumerate(subheadlines[:3], 1):  # Limit to first 3
            description_parts.append(f"  {i}. \"{subheadline}\"")
    
    # Section labels
    if section_labels:
        description_parts.append("")
        description_parts.append("Sections:")
        for section in section_labels[:5]:  # Limit to first 5
            description_parts.append(f"  - {section}")
    
    # Weather info
    if weather_info:
        description_parts.append("")
        description_parts.append(f"Weather: {weather_info}")
    
    # Website info
    if website_info:
        description_parts.append("")
        description_parts.append(f"Website: {website_info}")
    
    # Context
    description_parts.append("")
    description_parts.append("Context: This is a newspaper front page containing news articles, headlines, and images typical of a daily publication.")
    
    return "\n".join(description_parts)


def _extract_raw_text_from_results(results):
    """
    Extract raw text from PaddleOCR results.
    
    Args:
        results: PaddleOCR results (list or dict)
        
    Returns:
        str: Extracted raw text
    """
    text_elements = []
    all_text_sources = []
    
    if isinstance(results, list):
        for res in results:
            # Extract from parsing_res_list (most important source)
            if hasattr(res, 'get'):
                try:
                    parsing_res_list = res.get('parsing_res_list')
                    if parsing_res_list:
                        for parsing_res in parsing_res_list:
                            if isinstance(parsing_res, dict):
                                # Try all possible text keys (block_content is the main one)
                                for text_key in ['block_content', 'text', 'content', 'ocr_text', 'result', 'ocr_result', 'transcription']:
                                    text_val = parsing_res.get(text_key)
                                    if text_val and isinstance(text_val, str) and text_val.strip():
                                        all_text_sources.append(text_val.strip())
                            elif hasattr(parsing_res, '__class__'):
                                # Handle PaddleOCRVLBlock objects
                                for attr_name in ['block_content', 'content', 'text', 'ocr_text']:
                                    if hasattr(parsing_res, attr_name):
                                        try:
                                            text_val = getattr(parsing_res, attr_name)
                                            if text_val and isinstance(text_val, str) and text_val.strip():
                                                all_text_sources.append(text_val.strip())
                                                break
                                        except:
                                            pass
                            elif isinstance(parsing_res, str) and parsing_res.strip():
                                all_text_sources.append(parsing_res.strip())
                except Exception as e:
                    logger.debug(f"Error extracting from parsing_res_list: {e}")
            
            # Extract from markdown property
            if hasattr(res, 'markdown'):
                try:
                    md = res.markdown
                    if callable(md):
                        md = md()
                    if md and isinstance(md, str) and md.strip():
                        input_path = res.get('input_path', '') if hasattr(res, 'get') else ''
                        if md.strip() != input_path and len(md.strip()) > 10:
                            all_text_sources.append(md.strip())
                except Exception as e:
                    logger.debug(f"Error accessing markdown: {e}")
            
            # Extract from str property
            if hasattr(res, 'str'):
                try:
                    str_val = res.str
                    if callable(str_val):
                        str_val = str_val()
                    if str_val and isinstance(str_val, str) and str_val.strip():
                        input_path = res.get('input_path', '') if hasattr(res, 'get') else ''
                        if str_val.strip() != input_path and len(str_val.strip()) > 10:
                            all_text_sources.append(str_val.strip())
                except Exception as e:
                    logger.debug(f"Error accessing str: {e}")
    
    elif isinstance(results, dict):
        # Handle dictionary results
        for key in ['text', 'content', 'ocr_result', 'result', 'data']:
            if key in results:
                value = results[key]
                if isinstance(value, str) and value.strip():
                    all_text_sources.append(value.strip())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            all_text_sources.append(item.strip())
    
    elif isinstance(results, str):
        if results.strip():
            all_text_sources.append(results.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in all_text_sources:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    # Combine all unique texts
    if unique_texts:
        return '\n'.join(unique_texts)
    
    return ""


def extract_text(image, language_hints=None):
    """
    Extract text from a PIL Image using PaddleOCR.
    
    Args:
        image: PIL Image object
        language_hints: Optional list of language codes (not used by PaddleOCR, kept for API compatibility)
        
    Returns:
        tuple: (raw_text, structured_text, metadata)
            - raw_text: Extracted raw text for model prediction
            - structured_text: Formatted newspaper description (or empty string if not applicable)
            - metadata: Dictionary with OCR metadata
    """
    pipeline = _get_pipeline()
    
    # Save image to temporary file for PaddleOCR processing
    temp_file = None
    try:
        # Create temporary file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        
        # Save PIL image to temporary file
        image.save(temp_file, 'JPEG', quality=95)
        
        # Process the image
        logger.info("Processing image with PaddleOCR...")
        # Suppress PaddleOCR internal logging during prediction to avoid Tensor conversion errors
        import warnings
        import paddle
        
        # Try to enable dynamic mode if available
        try:
            if hasattr(paddle, 'enable_static'):
                # Disable static mode if possible
                pass
        except:
            pass
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Temporarily reduce logging level for all loggers
            old_levels = {}
            for logger_name in ['paddle', 'ppocr', 'paddlex', 'root']:
                logger_obj = logging.getLogger(logger_name)
                old_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.ERROR)
            
            try:
                results = pipeline.predict(temp_file)
            finally:
                # Restore logging levels
                for logger_name, old_level in old_levels.items():
                    logging.getLogger(logger_name).setLevel(old_level)
        
        logger.info("OCR prediction completed")
        
        # Extract parsing_res_list for structured formatting
        parsing_res_list = None
        if isinstance(results, list) and len(results) > 0:
            res = results[0]
            if hasattr(res, 'get'):
                parsing_res_list = res.get('parsing_res_list')
        
        # Extract raw text
        raw_text = _extract_raw_text_from_results(results)
        
        # Generate structured text if parsing_res_list is available
        structured_text = ""
        if parsing_res_list:
            try:
                structured_text = format_newspaper_structure(parsing_res_list)
            except Exception as e:
                logger.warning(f"Error generating structured text: {e}")
                structured_text = ""
        
        # Build metadata
        metadata = {
            'engine': 'paddleocr',
            'text_detections': len(raw_text.split('\n')) if raw_text else 0,
            'has_structured_format': bool(structured_text)
        }
        
        return raw_text, structured_text, metadata
        
    except Exception as e:
        error_msg = str(e)
        # Check if it's the Tensor conversion error
        if "int(Tensor)" in error_msg or "static graph mode" in error_msg:
            logger.warning("PaddleOCR Tensor conversion error detected. Retrying with minimal logging...")
            # Try once more with minimal logging - ensure temp_file and pipeline are still available
            if temp_file and os.path.exists(temp_file) and pipeline:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Set all relevant loggers to CRITICAL
                        for logger_name in ['paddle', 'ppocr', 'paddlex', 'root']:
                            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
                        
                        results = pipeline.predict(temp_file)
                        raw_text = _extract_raw_text_from_results(results)
                        
                        # Extract parsing_res_list
                        parsing_res_list = None
                        if isinstance(results, list) and len(results) > 0:
                            res = results[0]
                            if hasattr(res, 'get'):
                                parsing_res_list = res.get('parsing_res_list')
                        
                        structured_text = ""
                        if parsing_res_list:
                            try:
                                structured_text = format_newspaper_structure(parsing_res_list)
                            except:
                                structured_text = ""
                        
                        metadata = {
                            'engine': 'paddleocr',
                            'text_detections': len(raw_text.split('\n')) if raw_text else 0,
                            'has_structured_format': bool(structured_text)
                        }
                        
                        return raw_text, structured_text, metadata
                except Exception as retry_error:
                    logger.error(f"OCR extraction failed even after retry: {retry_error}")
                    raise Exception(f"OCR extraction failed: {str(retry_error)}")
        
        logger.error(f"OCR extraction failed: {e}")
        raise Exception(f"OCR extraction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")


def get_supported_languages():
    """
    Get list of supported OCR languages.
    Note: PaddleOCR supports many languages, but this is a simplified list.
    
    Returns:
        list: List of supported language codes
    """
    # PaddleOCR supports many languages, but we'll return a common subset
    # The actual language support depends on the installed PaddleOCR models
    return [
        'en',  # English
        'ch',  # Chinese
        'fr',  # French
        'de',  # German
        'es',  # Spanish
        'it',  # Italian
        'pt',  # Portuguese
        'ru',  # Russian
        'ja',  # Japanese
        'ko',  # Korean
    ]

