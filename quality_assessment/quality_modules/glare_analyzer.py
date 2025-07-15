"""
Glare and Overexposure Analysis Module
Detects glare, overexposure, and bright spots in document images
"""
import cv2
import numpy as np
from typing import Dict, List
from agno.utils.log import logger

EXPOSURE_THRESHOLDS = {
    'overexposed_pixel_value': 240,      # Pixel value considered overexposed
    'max_overexposed_percent': 5,        # Max acceptable % of overexposed pixels
    'bright_region_threshold': 220,      # Threshold for bright regions
    'glare_intensity_threshold': 250,    # Threshold for glare spots
    'histogram_peak_position': 200       # Position of histogram peak for overexposure
}


class GlareAnalyzer:
    
    def __init__(self):
        logger.info(f"🔍 ***** Starting Glare Analyzer...")
    
    def analyze_histogram(self, cv2_image: np.ndarray) -> Dict:
        """
        Analyze image histogram to detect overexposure
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with histogram analysis results
        """
        logger.debug(f"📊 Analyzing histogram for overexposure")
        
        if len(cv2_image.shape) == 3:
            gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2_image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        peak_value = np.argmax(hist)
        peak_intensity = hist[peak_value]
        
        total_pixels = gray.shape[0] * gray.shape[1]
        overexposed_pixels = np.sum(hist[EXPOSURE_THRESHOLDS['overexposed_pixel_value']:])
        overexposed_percent = (overexposed_pixels / total_pixels) * 100
        
        pixel_values = np.arange(256)
        mean_brightness = np.average(pixel_values, weights=hist)
        variance = np.average((pixel_values - mean_brightness) ** 2, weights=hist)
        std_deviation = np.sqrt(variance)
        
        is_overexposed = (
            peak_value > EXPOSURE_THRESHOLDS['histogram_peak_position'] and
            overexposed_percent > EXPOSURE_THRESHOLDS['max_overexposed_percent']
        )
        
        dark_pixels = np.sum(hist[:50])
        dark_pixels_percent = (dark_pixels / total_pixels) * 100
        
        return {
            'peak_position': int(peak_value),
            'mean_brightness': round(mean_brightness, 1),
            'std_deviation': round(std_deviation, 1),
            'overexposed_percent': round(overexposed_percent, 2),
            'dark_pixels_percent': round(dark_pixels_percent, 2),
            'is_overexposed': is_overexposed,
            'contrast_ratio': round(std_deviation / mean_brightness, 2) if mean_brightness > 0 else 0
        }
    
    def detect_glare_regions(self, cv2_image: np.ndarray) -> Dict:
        """
        Detect specific glare regions using HSV analysis
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with glare region information
        """
        logger.debug(f"🔦 Detecting glare regions")
        
        if len(cv2_image.shape) == 2:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        glare_mask = v > EXPOSURE_THRESHOLDS['glare_intensity_threshold']
        
        bright_mask = (v > EXPOSURE_THRESHOLDS['bright_region_threshold']) & (s < 30)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            glare_mask.astype(np.uint8), connectivity=8
        )
        
        min_area = 100  # minimum pixels for a glare region
        glare_regions = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                
                glare_regions.append({
                    'bbox': (x, y, w, h),
                    'center': (int(cx), int(cy)),
                    'area': area,
                    'intensity': np.mean(v[y:y+h, x:x+w])
                })
        
        total_glare_area = sum(region['area'] for region in glare_regions)
        total_image_area = cv2_image.shape[0] * cv2_image.shape[1]
        glare_coverage_percent = (total_glare_area / total_image_area) * 100
        
        return {
            'glare_regions': glare_regions,
            'num_glare_spots': len(glare_regions),
            'glare_coverage_percent': round(glare_coverage_percent, 2),
            'has_significant_glare': glare_coverage_percent > 2.0
        }
    
    def assess_glare(self, cv2_image: np.ndarray) -> Dict:
        """
        Comprehensive glare and exposure assessment
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary containing glare assessment results
        """
        logger.info(f"🔬 Assessing glare and exposure...")
        
        histogram_analysis = self.analyze_histogram(cv2_image)
        
        glare_detection = self.detect_glare_regions(cv2_image)
        
        glare_penalty = (
            histogram_analysis['overexposed_percent'] * 2 +  # Weight overexposure
            glare_detection['glare_coverage_percent'] * 3 +   # Weight glare coverage more
            (10 if histogram_analysis['is_overexposed'] else 0)  # Additional penalty for overexposure
        )
        
        glare_score = max(0, 100 - glare_penalty)
        
        if glare_score >= 95:
            glare_level = 'None'
        elif glare_score >= 85:
            glare_level = 'Minimal'
        elif glare_score >= 70:
            glare_level = 'Moderate'
        elif glare_score >= 50:
            glare_level = 'Significant'
        else:
            glare_level = 'Severe'
        
        glare_patterns = self._identify_glare_patterns(glare_detection['glare_regions'])
        
        results = {
            'exposure_metrics': {
                'mean_brightness': histogram_analysis['mean_brightness'],
                'overexposed_percent': histogram_analysis['overexposed_percent'],
                'is_overexposed': histogram_analysis['is_overexposed'],
                'contrast_ratio': histogram_analysis['contrast_ratio']
            },
            'glare_analysis': {
                'glare_score': round(glare_score, 1),
                'glare_level': glare_level,
                'num_glare_spots': glare_detection['num_glare_spots'],
                'glare_coverage_percent': glare_detection['glare_coverage_percent'],
                'glare_patterns': glare_patterns
            },
            'affected_regions': glare_detection['glare_regions'][:5],  # Top 5 largest glare spots
            'recommendations': self._generate_recommendations(glare_score, glare_level, histogram_analysis, glare_patterns)
        }
        
        logger.info(f"✅ ***** Glare analysis done. Score: {glare_score}/100")
        return results
    
    def _identify_glare_patterns(self, glare_regions: List[Dict]) -> Dict:
        """
        Identify specific glare patterns (e.g., flash reflection, window glare)
        
        Args:
            glare_regions: List of detected glare regions
            
        Returns:
            Dictionary describing glare patterns
        """
        if not glare_regions:
            return {'type': 'none', 'description': 'No glare detected'}
        
        sorted_regions = sorted(glare_regions, key=lambda x: x['area'], reverse=True)
        
        if len(sorted_regions) == 1 or (len(sorted_regions) > 0 and sorted_regions[0]['area'] > sum(r['area'] for r in sorted_regions[1:]) * 2):
            return {'type': 'flash', 'description': 'Camera flash reflection detected'}
        
        edge_glare = False
        for region in sorted_regions[:3]:
            x, y, w, h = region['bbox']
            if x < 50 or y < 50:  # Near top or left edge
                edge_glare = True
                break
        
        if edge_glare:
            return {'type': 'ambient', 'description': 'Edge lighting/window glare detected'}
        
        if len(glare_regions) > 5:
            return {'type': 'multiple', 'description': 'Multiple glare spots detected'}
        
        return {'type': 'general', 'description': 'General brightness issues'}
    
    def _generate_recommendations(self, glare_score: float, glare_level: str, 
                                histogram_analysis: Dict, glare_patterns: Dict) -> list:
        recommendations = []
        
        if histogram_analysis['is_overexposed']:
            recommendations.append("📷 Image is overexposed. Reduce camera exposure or lighting.")
        
        if glare_patterns['type'] == 'flash':
            recommendations.append("🔦 Disable camera flash to avoid reflections.")
        elif glare_patterns['type'] == 'ambient':
            recommendations.append("🪟 Reposition to avoid window/light reflections.")
        
        if glare_level in ['Significant', 'Severe']:
            recommendations.append("⚠️ Significant glare detected. Adjust lighting and recapture.")
        elif glare_level == 'Moderate':
            recommendations.append("💡 Moderate glare present. Consider adjusting angle or lighting.")
        
        if histogram_analysis['contrast_ratio'] < 0.2:
            recommendations.append("🎨 Low contrast detected. Ensure even lighting across document.")
        
        if not recommendations:
            recommendations.append("✅ Lighting and exposure are optimal.")
        
        return recommendations 