"""
Blur Detection Module
Detects image blur using Laplacian variance method
Based on: https://www.dynamsoft.com/codepool/quality-evaluation-of-scanned-document-images.html
"""
import cv2
import numpy as np
from typing import Dict, Tuple
from ..utils import convert_to_grayscale, calculate_score
from agno.utils.log import logger

BLUR_THRESHOLDS = {
    'sharp': 500,          # Very sharp image
    'acceptable': 200,     # Acceptable sharpness (industry standard)
    'slightly_blurry': 100,  # Slightly blurry but may be readable
    'very_blurry': 50      # Too blurry for reliable processing
}


class BlurDetector:
    
    def __init__(self):
        logger.info(f"🔍 ***** Starting Blur Detector...")
    
    def calculate_laplacian_variance(self, cv2_image: np.ndarray) -> float:
        """
        Calculate Laplacian variance to detect blur
        
        As pixels in a blurred image have similar neighboring pixels,
        we apply a 3x3 Laplacian kernel on the grayscale image
        and calculate the variance.
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Variance value (higher = sharper)
        """
        logger.debug(f"📊 Calculating Laplacian variance")
        
        gray = convert_to_grayscale(cv2_image)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        variance = laplacian.var()
        
        logger.debug(f"📈 Laplacian variance = {variance:.2f}")
        return variance
    
    def detect_motion_blur(self, cv2_image: np.ndarray) -> Dict:
        """
        Detect motion blur using kernel convolution
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with motion blur detection results
        """
        gray = convert_to_grayscale(cv2_image)
        
        kernel_sizes = [15, 21, 27]
        max_motion_score = 0
        dominant_direction = None
        
        for size in kernel_sizes:
            kernel_h = np.zeros((size, size))
            kernel_h[int((size-1)/2), :] = np.ones(size) / size
            motion_h = cv2.filter2D(gray, -1, kernel_h)
            diff_h = cv2.absdiff(gray, motion_h).mean()
            
            kernel_v = np.zeros((size, size))
            kernel_v[:, int((size-1)/2)] = np.ones(size) / size
            motion_v = cv2.filter2D(gray, -1, kernel_v)
            diff_v = cv2.absdiff(gray, motion_v).mean()
            
            kernel_d1 = np.eye(size) / size
            motion_d1 = cv2.filter2D(gray, -1, kernel_d1)
            diff_d1 = cv2.absdiff(gray, motion_d1).mean()
            
            kernel_d2 = np.fliplr(kernel_d1)
            motion_d2 = cv2.filter2D(gray, -1, kernel_d2)
            diff_d2 = cv2.absdiff(gray, motion_d2).mean()
            
            min_diff = min(diff_h, diff_v, diff_d1, diff_d2)
            if min_diff < max_motion_score or max_motion_score == 0:
                max_motion_score = min_diff
                if min_diff == diff_h:
                    dominant_direction = 'horizontal'
                elif min_diff == diff_v:
                    dominant_direction = 'vertical'
                elif min_diff == diff_d1:
                    dominant_direction = 'diagonal_1'
                else:
                    dominant_direction = 'diagonal_2'
        
        return {
            'detected': max_motion_score < 5,  # Threshold for motion blur
            'score': max_motion_score,
            'direction': dominant_direction
        }
    
    def assess_blur(self, cv2_image: np.ndarray) -> Dict:
        """
        Comprehensive blur assessment
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary containing blur assessment results
        """
        logger.info(f"🔬 Assessing image blur...")
        
        variance = self.calculate_laplacian_variance(cv2_image)
        
        motion_blur = self.detect_motion_blur(cv2_image)
        
        blur_score = calculate_score(
            variance,
            BLUR_THRESHOLDS['very_blurry'],
            BLUR_THRESHOLDS['sharp']
        )
        
        if variance >= BLUR_THRESHOLDS['sharp']:
            blur_level = 'Very Sharp'
            is_blurry = False
        elif variance >= BLUR_THRESHOLDS['acceptable']:
            blur_level = 'Sharp'
            is_blurry = False
        elif variance >= BLUR_THRESHOLDS['slightly_blurry']:
            blur_level = 'Slightly Blurry'
            is_blurry = True
        elif variance >= BLUR_THRESHOLDS['very_blurry']:
            blur_level = 'Blurry'
            is_blurry = True
        else:
            blur_level = 'Very Blurry'
            is_blurry = True
        
        focus_map = self._generate_focus_map(cv2_image)
        
        results = {
            'metrics': {
                'laplacian_variance': round(variance, 2),
                'is_blurry': is_blurry,
                'blur_score': blur_score,
                'blur_level': blur_level
            },
            'motion_blur': motion_blur,
            'focus_distribution': {
                'sharp_areas_percent': round(focus_map['sharp_percent'], 1),
                'uniform_sharpness': focus_map['uniform']
            },
            'recommendations': self._generate_recommendations(variance, blur_level, motion_blur)
        }
        
        logger.info(f"✅ ***** Blur detection done. Score: {blur_score}/100")
        return results
    
    def _generate_focus_map(self, cv2_image: np.ndarray) -> Dict:
        """
        Generate a map showing which areas of the image are in focus
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with focus distribution information
        """
        gray = convert_to_grayscale(cv2_image)
        
        h, w = gray.shape
        block_size = 50
        sharp_blocks = 0
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                variance = laplacian.var()
                
                if variance > BLUR_THRESHOLDS['acceptable']:
                    sharp_blocks += 1
                total_blocks += 1
        
        sharp_percent = (sharp_blocks / total_blocks) * 100 if total_blocks > 0 else 0
        uniform = sharp_percent > 80 or sharp_percent < 20  # Either mostly sharp or mostly blurry
        
        return {
            'sharp_percent': sharp_percent,
            'uniform': uniform
        }
    
    def _generate_recommendations(self, variance: float, blur_level: str, motion_blur: Dict) -> list:
        recommendations = []
        
        if variance < BLUR_THRESHOLDS['acceptable']:
            recommendations.append("🔄 Image is blurry. Ensure camera is focused and stable during capture.")
        
        if motion_blur['detected']:
            recommendations.append(f"📸 Motion blur detected ({motion_blur['direction']}). Use a tripod or scanner for better results.")
        
        if blur_level in ['Blurry', 'Very Blurry']:
            recommendations.append("⚠️ Image too blurry for reliable text extraction. Please recapture.")
        elif blur_level == 'Slightly Blurry':
            recommendations.append("📈 Image is slightly blurry. Consider recapturing for optimal results.")
        
        if not recommendations:
            recommendations.append("✅ Image sharpness is excellent.")
        
        return recommendations 