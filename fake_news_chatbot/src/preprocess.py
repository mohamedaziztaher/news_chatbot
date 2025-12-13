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

def preprocess_text(text):
    """
    Alias for clean_text() for convenience.
    
    Args:
        text (str): Raw text to preprocess
    
    Returns:
        str: Preprocessed text
    """
    return clean_text(text)

