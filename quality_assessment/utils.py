"""
Utility functions for image quality assessment
"""
import logging
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from agno.utils.log import logger

class ImageLoadError(Exception):
    pass


def load_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Load image using both OpenCV and PIL for comprehensive analysis
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (cv2_image, pil_image)
        
    Raises:
        ImageLoadError: If image cannot be loaded
    """
    logger.info(f"📸 Loading image from: {image_path}")
    
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f"Image file not found: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if path.suffix.lower() not in valid_extensions:
        raise ImageLoadError(f"Invalid image format: {path.suffix}")
    
    try:
        cv2_image = cv2.imread(str(path))
        if cv2_image is None:
            raise ImageLoadError(f"OpenCV failed to load image: {image_path}")
        
        pil_image = Image.open(str(path))
        
        logger.info(f"✅ Successfully loaded image: {cv2_image.shape}")
        return cv2_image, pil_image
    
    except Exception as e:
        logger.error(f"❌ Error loading image: {str(e)}")
        raise ImageLoadError(f"Failed to load image: {str(e)}")


def get_image_dimensions(cv2_image: np.ndarray) -> Tuple[int, int]:
    """
    Get image dimensions (width, height)
    
    Args:
        cv2_image: OpenCV image array
        
    Returns:
        Tuple of (width, height)
    """
    height, width = cv2_image.shape[:2]
    logger.debug(f"📏 Image dimensions: {width}x{height}")
    return width, height


def convert_to_grayscale(cv2_image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if not already
    
    Args:
        cv2_image: OpenCV image array
        
    Returns:
        Grayscale image
    """
    if len(cv2_image.shape) == 2:
        return cv2_image
    
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    logger.debug(f"🎨 Converted to grayscale")
    return gray


def calculate_score(value: float, min_acceptable: float, max_optimal: float, 
                   invert: bool = False) -> float:
    """
    Calculate a 0-100 score based on value ranges
    
    Args:
        value: The measured value
        min_acceptable: Minimum acceptable value
        max_optimal: Maximum optimal value
        invert: If True, lower values are better
        
    Returns:
        Score from 0 to 100
    """
    if invert:
        value = max_optimal - value + min_acceptable
    
    if value <= min_acceptable:
        return 0.0
    elif value >= max_optimal:
        return 100.0
    else:
        score = ((value - min_acceptable) / (max_optimal - min_acceptable)) * 100
        return round(score, 2)


def annotate_image(cv2_image: np.ndarray, issues: Dict[str, Any]) -> np.ndarray:
    """
    Annotate image with detected issues
    
    Args:
        cv2_image: OpenCV image array
        issues: Dictionary of detected issues with their locations
        
    Returns:
        Annotated image
    """
    annotated = cv2_image.copy()
    
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    for issue_type, details in issues.items():
        if 'regions' in details:
            for region in details['regions']:
                if 'bbox' in region:
                    x, y, w, h = region['bbox']
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(annotated, issue_type, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return annotated


def save_result_image(cv2_image: np.ndarray, output_path: str) -> None:
    """
    Save processed image to file
    
    Args:
        cv2_image: OpenCV image array
        output_path: Path to save the image
    """
    success = cv2.imwrite(output_path, cv2_image)
    if success:
        logger.info(f"💾 Saved result image to: {output_path}")
    else:
        logger.error(f"❌ Failed to save image to: {output_path}") 


def detect_image_type(cv2_image: np.ndarray, pil_image: Image.Image) -> Dict[str, Any]:
    """
    Detect whether an image is a digital screenshot or physical document scan
    
    Args:
        cv2_image: OpenCV image array
        pil_image: PIL Image object
        
    Returns:
        Dictionary with detection results
    """
    logger.info(f"🔍 Detecting image type (screenshot vs physical scan)...")
    
    height, width = cv2_image.shape[:2]
    
    # Check 1: Analyze color distribution
    if len(cv2_image.shape) == 3:
        reshaped = cv2_image.reshape(-1, cv2_image.shape[-1])
        unique_colors = len(np.unique(reshaped, axis=0))
        color_ratio = unique_colors / (width * height)
        
        is_low_color_variety = color_ratio < 0.1
    else:
        is_low_color_variety = False
    
    # Check 2: Look for common screenshot resolutions
    common_resolutions = [
        (1125, 2436),  # iPhone X/XS/11 Pro
        (1242, 2688),  # iPhone XS Max/11 Pro Max
        (1170, 2532),  # iPhone 12/13/14
        (1290, 2796),  # iPhone 14 Pro Max
        (1080, 1920),  # Common Android
        (1440, 2560),  # Common Android
        (1080, 2340),  # Common Android
        (1284, 2778),  # iPhone 12 Pro Max
    ]
    
    matches_common_resolution = any(
        (width == w and height == h) or (width == h and height == w) 
        for w, h in common_resolutions
    )
    
    # Check 3: Analyze edge characteristics
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY) if len(cv2_image.shape) == 3 else cv2_image
    edges = cv2.Canny(gray, 50, 150)
    
    edge_pixels = np.sum(edges > 0)
    total_pixels = width * height
    edge_density = edge_pixels / total_pixels
    
    has_regular_edges = edge_density < 0.05
    
    # Check 4: Analyze background uniformity
    corner_size = 50
    corners = [
        gray[:corner_size, :corner_size],  # Top-left
        gray[:corner_size, -corner_size:],  # Top-right
        gray[-corner_size:, :corner_size],  # Bottom-left
        gray[-corner_size:, -corner_size:],  # Bottom-right
    ]
    
    corner_stds = [np.std(corner) for corner in corners]
    has_uniform_background = all(std < 10 for std in corner_stds)
    
    # Check 5: Look for UI patterns (straight lines, rectangles)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    has_many_straight_lines = lines is not None and len(lines) > 20
    
    # Check 6: Analyze histogram characteristics
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()
    
    hist_entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    has_low_entropy = hist_entropy < 6.0
    
    screenshot_indicators = sum([
        is_low_color_variety,
        matches_common_resolution,
        has_regular_edges,
        has_uniform_background,
        has_many_straight_lines,
        has_low_entropy
    ])
    
    is_screenshot = screenshot_indicators >= 3
    confidence = screenshot_indicators / 6.0
    
    if is_screenshot:
        if has_uniform_background and has_many_straight_lines:
            image_subtype = "mobile_app_screenshot"
        elif matches_common_resolution:
            image_subtype = "device_screenshot"
        else:
            image_subtype = "digital_document"
    else:
        if width > 2000 or height > 2000:
            image_subtype = "high_res_scan"
        else:
            image_subtype = "photo_capture"
    
    result = {
        'is_digital_screenshot': is_screenshot,
        'confidence': round(confidence, 2),
        'image_subtype': image_subtype,
        'indicators': {
            'low_color_variety': is_low_color_variety,
            'matches_common_resolution': matches_common_resolution,
            'has_regular_edges': has_regular_edges,
            'has_uniform_background': has_uniform_background,
            'has_many_straight_lines': has_many_straight_lines,
            'low_histogram_entropy': has_low_entropy
        },
        'metadata': {
            'unique_colors': unique_colors if 'unique_colors' in locals() else None,
            'edge_density': round(edge_density, 3),
            'histogram_entropy': round(hist_entropy, 2)
        }
    }
    
    logger.info(f"📱 Image type: {image_subtype} (Screenshot: {is_screenshot}, Confidence: {confidence:.0%})")
    return result 