import re

def clean_text(text):
    """
    Clean and preprocess text by removing URLs, special characters, and converting to lowercase.
    
    Note: The trained model uses a Pipeline with TF-IDF vectorizer that handles preprocessing
    automatically. This function is provided for consistency with the training pipeline,
    but it's not strictly necessary to use it before prediction since the Pipeline handles
    text preprocessing internally.
    
    Args:
        text (str): Raw text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove special characters, keep only letters and spaces
    text = re.sub(r"[^A-Za-z ]", " ", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def preprocess_newspaper_text(text):
    """
    Preprocess OCR-extracted newspaper text to extract meaningful news content.
    
    This function is specifically designed for text extracted from newspaper images,
    which often contains metadata (dates, weather, prices, website URLs) mixed with
    actual news content (headlines, article snippets).
    
    Args:
        text (str): Raw OCR-extracted text from newspaper image
    
    Returns:
        str: Preprocessed text with meaningful news content
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    
    if not text.strip():
        return ""
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip common newspaper metadata patterns
        # Dates (e.g., "WEDNESDAY, AUGUST 25, 2010", "08/25/2010")
        if re.match(r'^(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)', line, re.IGNORECASE):
            continue
        if re.match(r'^[A-Z]+\s+\d{1,2},\s+\d{4}', line):
            continue
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', line):
            continue
        
        # Weather info (e.g., "AFTERNOON SHOWER - HIGH 80, LOW 66")
        if re.search(r'\b(HIGH|LOW|SHOWER|RAIN|SUNNY|CLOUDY|TEMPERATURE|°F|°C)\b', line, re.IGNORECASE):
            continue
        
        # Prices (e.g., "$1.00", "$2.50")
        if re.match(r'^\$?\d+\.?\d*\s*$', line):
            continue
        
        # Website URLs and domains (e.g., "washingtontimes.com")
        if re.search(r'\.(com|org|net|edu|gov|io|co)\b', line, re.IGNORECASE):
            # Keep if it's part of a longer sentence, remove if it's just the domain
            if len(line.split()) <= 2:
                continue
        
        # Single word section labels (e.g., "Economy", "Sports", "Politics")
        # Keep if it's part of a headline or article, skip if standalone
        if len(line.split()) == 1 and line.isupper() and len(line) < 20:
            continue
        
        # Keep newspaper names - they can be useful context for fake news detection
        # (We don't skip newspaper names anymore)
        
        # Keep meaningful content (headlines, article snippets)
        if len(line) > 10:  # Minimum length to be considered meaningful
            processed_lines.append(line)
    
    # Combine meaningful lines with better structure
    # Try to preserve sentence-like structure by joining with periods
    # This helps the model understand context better
    if len(processed_lines) > 1:
        # Join headlines and snippets with periods to create sentence-like structure
        processed_text = '. '.join(processed_lines)
    else:
        processed_text = ' '.join(processed_lines)
    
    # Apply standard cleaning (but preserve periods we just added)
    # First, do basic cleaning
    processed_text = re.sub(r"http\S+", "", processed_text)
    # Keep periods and basic punctuation for sentence structure
    processed_text = re.sub(r"[^A-Za-z .]", " ", processed_text)
    # Convert to lowercase
    processed_text = processed_text.lower()
    # Clean up multiple spaces but preserve sentence structure
    processed_text = re.sub(r'\s+', ' ', processed_text)
    # Clean up spaces around periods
    processed_text = re.sub(r'\s*\.\s*', '. ', processed_text)
    # Remove trailing spaces
    processed_text = processed_text.strip()
    
    return processed_text

def preprocess_text(text, is_newspaper_ocr=False):
    """
    Preprocess text for fake news prediction.
    
    Args:
        text (str): Raw text to preprocess
        is_newspaper_ocr (bool): If True, applies newspaper-specific preprocessing
    
    Returns:
        str: Preprocessed text
    """
    if is_newspaper_ocr:
        return preprocess_newspaper_text(text)
    else:
        return clean_text(text)

