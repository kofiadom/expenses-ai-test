"""
Resolution Quality Checker Module
Evaluates the resolution and DPI of document images
"""
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from ..utils import calculate_score, get_image_dimensions
from agno.utils.log import logger

DOCUMENT_SIZES = {
    'receipt': (3.125, 8.5),     # Standard receipt width
    'a4': (8.27, 11.69),          # A4 paper
    'letter': (8.5, 11),          # US Letter
    'id_card': (3.37, 2.125),     # Standard ID card
    'default': (8.5, 11)          # Default to letter size
}

DPI_THRESHOLDS = {
    'min_acceptable': 150,  # Absolute minimum for basic readability
    'recommended': 300,     # Recommended for good OCR accuracy
    'optimal': 600         # Optimal for highest quality
}


class ResolutionChecker:
    
    def __init__(self, document_type: str = 'default'):
        """
        Initialize resolution checker
        
        Args:
            document_type: Type of document ('receipt', 'a4', 'letter', 'id_card', 'default')
        """
        self.document_type = document_type
        self.expected_size = DOCUMENT_SIZES.get(document_type, DOCUMENT_SIZES['default'])
        logger.info(f"🔍 ***** Starting Resolution Checker for {document_type} documents...")
    
    def calculate_dpi(self, pil_image: Image.Image) -> Tuple[float, float]:
        """
        Calculate DPI from image metadata or estimate from dimensions
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Tuple of (horizontal_dpi, vertical_dpi)
        """
        logger.debug(f"📐 Calculating DPI for image")
        
        try:
            dpi = pil_image.info.get('dpi', None)
            if dpi and isinstance(dpi, tuple) and all(d > 0 for d in dpi):
                logger.debug(f"📊 Found DPI in metadata: {dpi}")
                return dpi
        except Exception as e:
            logger.debug(f"⚠️ Could not extract DPI from metadata: {e}")
        
        width, height = pil_image.size
        expected_width_inches, expected_height_inches = self.expected_size
        
        estimated_dpi_horizontal = width / expected_width_inches
        estimated_dpi_vertical = height / expected_height_inches
        
        avg_dpi = (estimated_dpi_horizontal + estimated_dpi_vertical) / 2
        
        logger.debug(f"📊 Estimated DPI: ({estimated_dpi_horizontal:.0f}, {estimated_dpi_vertical:.0f})")
        return (estimated_dpi_horizontal, estimated_dpi_vertical)
    
    def assess_resolution(self, cv2_image: np.ndarray, pil_image: Image.Image) -> Dict:
        """
        Assess the resolution quality of an image
        
        Args:
            cv2_image: OpenCV image array
            pil_image: PIL Image object
            
        Returns:
            Dictionary containing assessment results
        """
        logger.info(f"🔬 Assessing resolution quality...")
        
        width, height = get_image_dimensions(cv2_image)
        
        dpi_h, dpi_v = self.calculate_dpi(pil_image)
        avg_dpi = (dpi_h + dpi_v) / 2
        
        megapixels = (width * height) / 1_000_000
        
        dpi_score = calculate_score(
            avg_dpi, 
            DPI_THRESHOLDS['min_acceptable'], 
            DPI_THRESHOLDS['optimal']
        )
        
        if avg_dpi >= DPI_THRESHOLDS['optimal']:
            quality_level = 'Excellent'
        elif avg_dpi >= DPI_THRESHOLDS['recommended']:
            quality_level = 'Good'
        elif avg_dpi >= DPI_THRESHOLDS['min_acceptable']:
            quality_level = 'Fair'
        else:
            quality_level = 'Poor'
        
        actual_aspect_ratio = width / height
        expected_aspect_ratio = self.expected_size[0] / self.expected_size[1]
        aspect_ratio_deviation = abs(actual_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio * 100
        
        results = {
            'dimensions': {
                'width': width,
                'height': height,
                'megapixels': round(megapixels, 2)
            },
            'dpi': {
                'horizontal': round(dpi_h, 0),
                'vertical': round(dpi_v, 0),
                'average': round(avg_dpi, 0)
            },
            'quality': {
                'score': dpi_score,
                'level': quality_level,
                'meets_ocr_requirements': avg_dpi >= DPI_THRESHOLDS['recommended']
            },
            'aspect_ratio': {
                'actual': round(actual_aspect_ratio, 2),
                'expected': round(expected_aspect_ratio, 2),
                'deviation_percent': round(aspect_ratio_deviation, 1)
            },
            'recommendations': self._generate_recommendations(avg_dpi, quality_level)
        }
        
        logger.info(f"✅ ***** Resolution check done. Score: {dpi_score}/100")
        return results
    
    def _generate_recommendations(self, avg_dpi: float, quality_level: str) -> list:
        recommendations = []
        
        if avg_dpi < DPI_THRESHOLDS['min_acceptable']:
            recommendations.append("⚠️ Resolution too low for reliable OCR. Rescan at higher resolution.")
        elif avg_dpi < DPI_THRESHOLDS['recommended']:
            recommendations.append("📈 Consider rescanning at 300 DPI or higher for better OCR accuracy.")
        
        if quality_level == 'Poor':
            recommendations.append("🔄 Image quality insufficient. Please capture again with better camera/scanner settings.")
        
        if not recommendations:
            recommendations.append("✅ Resolution quality is excellent for document processing.")
        
        return recommendations 