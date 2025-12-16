"""
Image utility functions for handling image processing and validation.
"""

import base64
import io
from PIL import Image


def base64_to_image(base64_string):
    """
    Convert a base64-encoded string to a PIL Image object.
    
    Args:
        base64_string: Base64-encoded image string (with or without data URI prefix)
        
    Returns:
        PIL.Image: Image object
        
    Raises:
        ValueError: If the base64 string is invalid or image cannot be decoded
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,...")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def validate_image(image):
    """
    Validate that an image is in a supported format and has valid properties.
    
    Args:
        image: PIL Image object
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    if not isinstance(image, Image.Image):
        return False
    
    # Check if image has valid size
    if image.size[0] <= 0 or image.size[1] <= 0:
        return False
    
    # Check if image format is supported
    # PIL supports many formats, but we'll check for common ones
    try:
        # Check if image has valid mode
        if image.mode not in ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F']:
            return False
    except Exception:
        return False
    
    return True

