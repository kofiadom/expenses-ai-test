"""
Document Completeness Checker Module
Verifies document boundaries and detects missing edges/corners using edge detection
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils import convert_to_grayscale, calculate_score
from agno.utils.log import logger

EDGE_THRESHOLDS = {
    'canny_lower': 50,           # Lower threshold for Canny edge detection
    'canny_upper': 150,          # Upper threshold for Canny edge detection
    'min_contour_area': 1000,    # Minimum contour area to consider
    'corner_tolerance': 0.02,    # Tolerance for corner detection (2% of perimeter)
    'edge_gap_threshold': 50     # Maximum allowed gap in edge pixels
}


class CompletenessChecker:
    
    def __init__(self):
        logger.info(f"🔍 ***** Starting Completeness Checker...")
    
    def detect_document_edges(self, cv2_image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect document edges using Canny edge detection
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Tuple of (edge_image, contours)
        """
        logger.debug(f"🔲 Detecting document edges")
        
        gray = convert_to_grayscale(cv2_image)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 
                         EDGE_THRESHOLDS['canny_lower'], 
                         EDGE_THRESHOLDS['canny_upper'])
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return edges, contours
    
    def find_document_boundary(self, contours: List, image_shape: Tuple) -> Optional[np.ndarray]:
        """
        Find the main document boundary from contours
        
        Args:
            contours: List of contours
            image_shape: Shape of the image (height, width)
            
        Returns:
            Approximated document boundary or None
        """
        logger.debug(f"📐 Finding document boundary")
        
        h, w = image_shape[:2]
        min_area = (h * w) * 0.1  # Document should be at least 10% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            epsilon = EDGE_THRESHOLDS['corner_tolerance'] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4 and len(approx) <= 8:
                return approx
        
        return None
    
    def check_edge_completeness(self, edges: np.ndarray, boundary: np.ndarray) -> Dict:
        """
        Check if document edges are complete without gaps
        
        Args:
            edges: Edge detection result
            boundary: Document boundary points
            
        Returns:
            Dictionary with edge completeness analysis
        """
        logger.debug(f"🔍 Checking edge completeness")
        
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.drawContours(mask, [boundary], -1, 255, 2)
        
        boundary_pixels = np.sum(mask > 0)
        edge_pixels_on_boundary = np.sum((edges > 0) & (mask > 0))
        edge_coverage = (edge_pixels_on_boundary / boundary_pixels) * 100 if boundary_pixels > 0 else 0
        
        gaps = []
        boundary_points = boundary.reshape(-1, 2)
        
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            
            num_samples = int(np.linalg.norm(p2 - p1))
            if num_samples > 0:
                samples = np.linspace(p1, p2, num_samples)
                gap_length = 0
                
                for point in samples:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                        if edges[y, x] == 0:
                            gap_length += 1
                        else:
                            if gap_length > EDGE_THRESHOLDS['edge_gap_threshold']:
                                gaps.append({
                                    'start': (x - gap_length, y),
                                    'length': gap_length
                                })
                            gap_length = 0
        
        return {
            'edge_coverage_percent': round(edge_coverage, 1),
            'num_gaps': len(gaps),
            'has_complete_edges': edge_coverage > 80 and len(gaps) < 2
        }
    
    def check_corners(self, boundary: np.ndarray, image_shape: Tuple) -> Dict:
        """
        Check if all document corners are visible
        
        Args:
            boundary: Document boundary points
            image_shape: Shape of the image
            
        Returns:
            Dictionary with corner analysis
        """
        logger.debug(f"📍 Checking document corners")
        
        h, w = image_shape[:2]
        margin = 20  # Pixels from edge
        
        points = boundary.reshape(-1, 2)
        
        # Method 1: Extreme points
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        top_right = points[np.argmin(-points[:, 0] + points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmax(-points[:, 0] + points[:, 1])]
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
        
        missing_corners = []
        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        
        for i, (corner, name) in enumerate(zip(corners, corner_names)):
            x, y = corner
            is_cut_off = False
            
            if name == 'top_left' and (x < margin or y < margin):
                is_cut_off = True
            elif name == 'top_right' and (x > w - margin or y < margin):
                is_cut_off = True
            elif name == 'bottom_right' and (x > w - margin or y > h - margin):
                is_cut_off = True
            elif name == 'bottom_left' and (x < margin or y > h - margin):
                is_cut_off = True
            
            if is_cut_off:
                missing_corners.append(name)
        
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            angles.append(np.degrees(angle))
        
        angle_deviations = [abs(angle - 90) for angle in angles]
        is_rectangular = all(dev < 15 for dev in angle_deviations)  # 15-degree tolerance
        
        return {
            'detected_corners': 4 - len(missing_corners),
            'missing_corners': missing_corners,
            'all_corners_visible': len(missing_corners) == 0,
            'is_rectangular': is_rectangular,
            'corner_angles': [round(angle, 1) for angle in angles]
        }
    
    def assess_completeness(self, cv2_image: np.ndarray) -> Dict:
        """
        Comprehensive document completeness assessment
        
        Args:
            cv2_image: OpenCV image array
            
        Returns:
            Dictionary containing completeness assessment results
        """
        logger.info(f"🔬 Assessing document completeness...")
        
        edges, contours = self.detect_document_edges(cv2_image)
        
        boundary = self.find_document_boundary(contours, cv2_image.shape)
        
        if boundary is None:
            logger.warning(f"⚠️ No clear document boundary detected")
            return {
                'boundary_detected': False,
                'completeness_score': 0,
                'completeness_level': 'Cannot Detect',
                'issues': ['No clear document boundary detected'],
                'recommendations': ['🔄 Ensure entire document is visible and well-lit', 
                                  '📸 Place document on contrasting background']
            }
        
        edge_analysis = self.check_edge_completeness(edges, boundary)
        
        corner_analysis = self.check_corners(boundary, cv2_image.shape)
        
        score_components = [
            edge_analysis['edge_coverage_percent'] * 0.4,  # 40% weight
            (corner_analysis['detected_corners'] / 4) * 100 * 0.4,  # 40% weight
            (100 if corner_analysis['is_rectangular'] else 50) * 0.2  # 20% weight
        ]
        completeness_score = sum(score_components)
        
        if completeness_score >= 95:
            completeness_level = 'Complete'
        elif completeness_score >= 85:
            completeness_level = 'Nearly Complete'
        elif completeness_score >= 70:
            completeness_level = 'Partially Complete'
        else:
            completeness_level = 'Incomplete'
        
        issues = []
        if not edge_analysis['has_complete_edges']:
            issues.append(f"Edge gaps detected ({edge_analysis['num_gaps']} gaps)")
        if not corner_analysis['all_corners_visible']:
            issues.append(f"Missing corners: {', '.join(corner_analysis['missing_corners'])}")
        if not corner_analysis['is_rectangular']:
            issues.append("Document appears skewed or distorted")
        
        results = {
            'boundary_detected': True,
            'completeness_score': round(completeness_score, 1),
            'completeness_level': completeness_level,
            'edge_analysis': {
                'edge_coverage': edge_analysis['edge_coverage_percent'],
                'has_gaps': edge_analysis['num_gaps'] > 0,
                'num_gaps': edge_analysis['num_gaps']
            },
            'corner_analysis': {
                'visible_corners': corner_analysis['detected_corners'],
                'missing_corners': corner_analysis['missing_corners'],
                'is_rectangular': corner_analysis['is_rectangular']
            },
            'issues': issues,
            'boundary_points': boundary.tolist(),  # For visualization
            'recommendations': self._generate_recommendations(completeness_score, issues, corner_analysis)
        }
        
        logger.info(f"✅ ***** Completeness check done. Score: {completeness_score}/100")
        return results
    
    def _generate_recommendations(self, score: float, issues: List[str], corner_analysis: Dict) -> list:
        recommendations = []
        
        if score < 70:
            recommendations.append("⚠️ Document appears incomplete. Ensure entire document is in frame.")
        
        if corner_analysis['missing_corners']:
            recommendations.append("📐 Reposition document to include all corners.")
        
        if not corner_analysis['is_rectangular']:
            recommendations.append("📷 Capture document from directly above to avoid distortion.")
        
        if any('gaps' in issue for issue in issues):
            recommendations.append("🔦 Improve lighting to ensure clear edge detection.")
        
        if score >= 95:
            recommendations.append("✅ Document boundaries are complete and well-captured.")
        
        return recommendations 