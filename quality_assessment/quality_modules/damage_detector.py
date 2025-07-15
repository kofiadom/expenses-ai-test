"""
Document Damage Detection Module
Detects various types of physical damage: stains, tears, and folds
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
from ..utils import convert_to_grayscale
from agno.utils.log import logger

DAMAGE_THRESHOLDS = {
    'stain_color_variance': 30,        # Color variance threshold for stain detection
    'stain_min_area': 100,             # Minimum area for stain blob
    'stain_contrast_threshold': 0.15,  # Minimum contrast difference for stain
    
    'tear_edge_irregularity': 0.3,    # Irregularity threshold for tear detection
    'tear_min_length': 50,             # Minimum length for tear edge
    'tear_angle_deviation': 45,        # Max angle deviation for natural edges
    
    'fold_line_threshold': 100,        # Minimum line length for fold
    'fold_shadow_gradient': 20,        # Gradient threshold for fold shadow
    'fold_angle_tolerance': 10         # Angle tolerance for parallel folds
}


class DamageDetector:
    
    def __init__(self):
        logger.info(f"🔍 ***** Starting Damage Detector...")
    
    def detect_stains(self, cv2_image: np.ndarray) -> Dict:
        """
        Detect stains and discoloration using color variance analysis
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with stain detection results
        """
        logger.debug(f"🔍 Detecting stains and discoloration")
        
        if len(cv2_image.shape) == 2:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
        
        lab = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        kernel_size = 15
        
        mean_filter = cv2.blur(lab, (kernel_size, kernel_size))
        squared_filter = cv2.blur(lab ** 2, (kernel_size, kernel_size))
        variance = squared_filter - mean_filter ** 2
        
        total_variance = np.sqrt(np.sum(variance, axis=2))
        
        stain_mask = total_variance > DAMAGE_THRESHOLDS['stain_color_variance']
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stain_mask = cv2.morphologyEx(stain_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stains = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < DAMAGE_THRESHOLDS['stain_min_area']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            stain_region = lab[y:y+h, x:x+w]
            background_region = lab  # Simplified - in practice, would sample around stain
            
            stain_mean = np.mean(stain_region, axis=(0, 1))
            background_mean = np.mean(background_region, axis=(0, 1))
            color_diff = np.linalg.norm(stain_mean - background_mean)
            
            stain_intensity = np.mean(l_channel[y:y+h, x:x+w])
            background_intensity = np.mean(l_channel)
            contrast = abs(stain_intensity - background_intensity) / 255
            
            if contrast > DAMAGE_THRESHOLDS['stain_contrast_threshold']:
                stains.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contrast': round(contrast, 3),
                    'color_difference': round(color_diff, 1),
                    'type': self._classify_stain_type(color_diff, contrast, stain_mean)
                })
        
        total_stain_area = sum(stain['area'] for stain in stains)
        image_area = cv2_image.shape[0] * cv2_image.shape[1]
        stain_coverage = (total_stain_area / image_area) * 100
        
        return {
            'stains_detected': len(stains),
            'stain_regions': stains[:10],  # Top 10 largest stains
            'stain_coverage_percent': round(stain_coverage, 2),
            'has_significant_stains': stain_coverage > 1.0 or len(stains) > 5
        }
    
    def detect_tears(self, cv2_image: np.ndarray) -> Dict:
        """
        Detect tears using edge irregularity analysis
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with tear detection results
        """
        logger.debug(f"🔍 Detecting tears and rips")
        
        gray = convert_to_grayscale(cv2_image)
        
        edges = cv2.Canny(gray, 30, 100)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tears = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, False)
            if perimeter < DAMAGE_THRESHOLDS['tear_min_length']:
                continue
            
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            approx_perimeter = cv2.arcLength(approx, False)
            irregularity = 1 - (approx_perimeter / perimeter) if perimeter > 0 else 0
            
            if irregularity > DAMAGE_THRESHOLDS['tear_edge_irregularity']:
                x, y, w, h = cv2.boundingRect(contour)
                
                edge_angles = self._calculate_edge_angles(contour)
                angle_variance = np.var(edge_angles) if len(edge_angles) > 0 else 0
                
                if angle_variance > DAMAGE_THRESHOLDS['tear_angle_deviation']:
                    tears.append({
                        'bbox': (x, y, w, h),
                        'length': perimeter,
                        'irregularity': round(irregularity, 3),
                        'angle_variance': round(angle_variance, 1),
                        'severity': self._classify_tear_severity(irregularity, perimeter)
                    })
        
        return {
            'tears_detected': len(tears),
            'tear_regions': tears[:5],  # Top 5 most significant tears
            'has_tears': len(tears) > 0,
            'max_tear_length': max([t['length'] for t in tears], default=0)
        }
    
    def detect_folds(self, cv2_image: np.ndarray) -> Dict:
        """
        Detect folds using line detection and shadow analysis
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary with fold detection results
        """
        logger.debug(f"🔍 Detecting folds and creases")
        
        gray = convert_to_grayscale(cv2_image)
        
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        edges = cv2.Canny(filtered, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=DAMAGE_THRESHOLDS['fold_line_threshold'], 
                               maxLineGap=10)
        
        folds = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                gradient_strength = self._analyze_fold_gradient(gray, (x1, y1), (x2, y2))
                
                if gradient_strength > DAMAGE_THRESHOLDS['fold_shadow_gradient']:
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                    
                    folds.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': round(angle, 1),
                        'gradient_strength': round(gradient_strength, 1),
                        'type': self._classify_fold_type(angle, length)
                    })
        
        fold_groups = self._group_parallel_folds(folds)
        
        return {
            'folds_detected': len(folds),
            'fold_lines': folds[:10],  # Top 10 most prominent folds
            'fold_groups': len(fold_groups),
            'has_significant_folds': len(folds) > 2 or any(f['gradient_strength'] > 50 for f in folds),
            'fold_pattern': self._analyze_fold_pattern(fold_groups)
        }
    
    def assess_damage(self, cv2_image: np.ndarray) -> Dict:
        """
        Comprehensive damage assessment
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary containing damage assessment results
        """
        logger.info(f"🔬 Assessing document damage...")
        
        stain_results = self.detect_stains(cv2_image)
        tear_results = self.detect_tears(cv2_image)
        fold_results = self.detect_folds(cv2_image)
        
        stain_score = 100 - min(stain_results['stain_coverage_percent'] * 10, 50)
        tear_score = 100 - min(tear_results['tears_detected'] * 20, 100)
        fold_score = 100 - min(fold_results['folds_detected'] * 10, 60)
        
        damage_score = (stain_score * 0.4 + tear_score * 0.4 + fold_score * 0.2)
        
        if damage_score >= 95:
            damage_level = 'Pristine'
        elif damage_score >= 85:
            damage_level = 'Minor Damage'
        elif damage_score >= 70:
            damage_level = 'Moderate Damage'
        elif damage_score >= 50:
            damage_level = 'Significant Damage'
        else:
            damage_level = 'Severe Damage'
        
        damage_types = []
        if stain_results['has_significant_stains']:
            damage_types.append('stains')
        if tear_results['has_tears']:
            damage_types.append('tears')
        if fold_results['has_significant_folds']:
            damage_types.append('folds')
        
        results = {
            'damage_score': round(damage_score, 1),
            'damage_level': damage_level,
            'damage_types': damage_types,
            'stain_analysis': {
                'count': stain_results['stains_detected'],
                'coverage_percent': stain_results['stain_coverage_percent'],
                'regions': stain_results['stain_regions'][:3]  # Top 3
            },
            'tear_analysis': {
                'count': tear_results['tears_detected'],
                'max_length': tear_results['max_tear_length'],
                'regions': tear_results['tear_regions'][:3]  # Top 3
            },
            'fold_analysis': {
                'count': fold_results['folds_detected'],
                'pattern': fold_results['fold_pattern'],
                'lines': fold_results['fold_lines'][:3]  # Top 3
            },
            'recommendations': self._generate_recommendations(damage_score, damage_types, 
                                                            stain_results, tear_results, fold_results)
        }
        
        logger.info(f"✅ ***** Damage assessment done. Score: {damage_score}/100")
        return results
    
    def _classify_stain_type(self, color_diff: float, contrast: float, stain_color: np.ndarray) -> str:
        l, a, b = stain_color
        
        if contrast > 0.5:
            return 'ink_spill'
        elif l < 100 and abs(a) < 10 and abs(b) < 10:
            return 'dark_spot'
        elif b > 20:
            return 'yellowing'
        elif a > 20:
            return 'reddish_stain'
        else:
            return 'general_discoloration'
    
    def _calculate_edge_angles(self, contour: np.ndarray) -> List[float]:
        angles = []
        points = contour.squeeze()
        
        if len(points) < 3:
            return angles
        
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            v1 = p1 - p2
            v2 = p3 - p2
            
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(np.degrees(angle))
        
        return angles
    
    def _classify_tear_severity(self, irregularity: float, length: float) -> str:
        if irregularity > 0.6 and length > 200:
            return 'severe'
        elif irregularity > 0.4 or length > 100:
            return 'moderate'
        else:
            return 'minor'
    
    def _analyze_fold_gradient(self, gray_image: np.ndarray, start: Tuple, end: Tuple) -> float:
        x1, y1 = start
        x2, y2 = end
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return 0
        
        perp_x = -dy / length
        perp_y = dx / length
        
        samples = []
        sample_dist = 10
        
        for t in np.linspace(0, 1, 10):
            cx = int(x1 + t * dx)
            cy = int(y1 + t * dy)
            
            for side in [-1, 1]:
                sx = int(cx + side * sample_dist * perp_x)
                sy = int(cy + side * sample_dist * perp_y)
                
                if 0 <= sx < gray_image.shape[1] and 0 <= sy < gray_image.shape[0]:
                    samples.append(gray_image[sy, sx])
        
        if len(samples) < 2:
            return 0
        
        return np.std(samples)
    
    def _classify_fold_type(self, angle: float, length: float) -> str:
        abs_angle = abs(angle)
        
        if abs_angle < 10 or abs_angle > 170:
            return 'horizontal_fold'
        elif 80 < abs_angle < 100:
            return 'vertical_fold'
        elif length > 300:
            return 'major_crease'
        else:
            return 'minor_crease'
    
    def _group_parallel_folds(self, folds: List[Dict]) -> List[List[Dict]]:
        if not folds:
            return []
        
        groups = []
        used = set()
        
        for i, fold1 in enumerate(folds):
            if i in used:
                continue
            
            group = [fold1]
            used.add(i)
            
            for j, fold2 in enumerate(folds[i+1:], i+1):
                if j in used:
                    continue
                
                angle_diff = abs(fold1['angle'] - fold2['angle'])
                if angle_diff < DAMAGE_THRESHOLDS['fold_angle_tolerance']:
                    group.append(fold2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _analyze_fold_pattern(self, fold_groups: List[List[Dict]]) -> str:
        if not fold_groups:
            return 'none'
        elif len(fold_groups) == 1 and len(fold_groups[0]) > 2:
            return 'accordion_fold'
        elif len(fold_groups) == 2 and all(len(g) == 1 for g in fold_groups):
            angles = [g[0]['angle'] for g in fold_groups]
            if abs(abs(angles[0] - angles[1]) - 90) < 20:
                return 'cross_fold'
        elif any(len(g) > 1 for g in fold_groups):
            return 'multiple_parallel'
        else:
            return 'irregular'
    
    def _generate_recommendations(self, damage_score: float, damage_types: List[str],
                                stain_results: Dict, tear_results: Dict, fold_results: Dict) -> list:
        recommendations = []
        
        if 'stains' in damage_types:
            if any(s['type'] == 'ink_spill' for s in stain_results['stain_regions']):
                recommendations.append("🧹 Ink spills detected. Consider professional document cleaning.")
            else:
                recommendations.append("🧼 Stains detected. Clean document surface before scanning.")
        
        if 'tears' in damage_types:
            if any(t['severity'] == 'severe' for t in tear_results['tear_regions']):
                recommendations.append("🩹 Severe tears detected. Use document tape on back before scanning.")
            else:
                recommendations.append("📎 Minor tears detected. Handle document carefully.")
        
        if 'folds' in damage_types:
            if fold_results['fold_pattern'] == 'accordion_fold':
                recommendations.append("📐 Accordion folds detected. Flatten document under weight before scanning.")
            else:
                recommendations.append("🗞️ Folds detected. Iron or press document flat if possible.")
        
        if damage_score < 50:
            recommendations.append("⚠️ Significant damage detected. Consider document restoration.")
        elif damage_score >= 95:
            recommendations.append("✅ Document is in excellent physical condition.")
        
        return recommendations 