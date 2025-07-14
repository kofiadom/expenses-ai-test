import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, measure, morphology
from typing import Dict, Tuple
from imutils.perspective import four_point_transform


class ReceiptDamageAnalyzer:
    """
    Advanced damage analysis for receipt images to assess OCR suitability.
    Detects folds, tears, stains, and other quality issues.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.damage_threshold = {
            'folds': 0.15,      # 15% fold coverage threshold
            'tears': 0.08,      # 8% tear coverage threshold  
            'stains': 0.12,     # 12% stain coverage threshold
            'overall': 0.25     # 25% overall damage threshold
        }

    def extract_receipt_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop the receipt from the input image.
        Returns the cropped receipt image, or None if not found.
        """
        image_resized = imutils.resize(image, width=500)
        ratio = image.shape[1] / float(image_resized.shape[1])
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        cnts = cv2.findContours(adaptive_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return four_point_transform(image, approx.reshape(4, 2) * ratio)
        return None

    def analyze_receipt_damage(self, image_path: str) -> Dict:
        """
        Main function to analyze receipt damage.
        Now extracts the receipt region before analysis.
        
        Args:
            image_path: Path to the receipt image
            
        Returns:
            Dictionary containing damage analysis results
        """
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        original_image = image.copy()  # Save original for visualization

        # --- Receipt extraction step ---
        receipt_img = self.extract_receipt_from_image(image)
        if receipt_img is not None:
            image = receipt_img  # Use cropped receipt for analysis
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Resize if too large (for performance)
        height, width = gray.shape
        if height > 2000 or width > 2000:
            scale = min(2000/height, 2000/width)
            new_height, new_width = int(height * scale), int(width * scale)
            gray = cv2.resize(gray, (new_width, new_height))
            hsv = cv2.resize(hsv, (new_width, new_height))
            image = cv2.resize(image, (new_width, new_height))
        
        # Perform damage analysis
        fold_analysis, fold_mask = self._detect_folds(gray)
        tear_analysis, tear_mask = self._detect_tears(gray)
        stain_analysis, stain_mask = self._detect_stains(image, hsv)
        contrast_analysis = self._analyze_contrast(gray)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            fold_analysis, tear_analysis, stain_analysis, contrast_analysis
        )
        
        # Determine OCR suitability
        ocr_suitable = self._assess_ocr_suitability(overall_score)
        
        results = {
            'overall_score': overall_score,
            'ocr_suitable': ocr_suitable,
            'damage_details': {
                'folds': fold_analysis,
                'tears': tear_analysis,
                'stains': stain_analysis,
                'contrast': contrast_analysis
            }
        }
        
        if self.debug_mode:
            self._create_debug_visualization(
                original_image, image, gray, results, fold_mask, tear_mask, stain_mask
            )

        return results
    
    def _detect_folds(self, gray: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Detect folds and creases in the receipt."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 30, 100)
        
        # Use HoughLines to detect fold lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Detect shadows and highlights that indicate folds
        # Apply morphological operations to find linear patterns
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal and vertical fold patterns
        horizontal_folds = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)
        vertical_folds = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Combine fold detections
        fold_mask = (horizontal_folds > 0) | (vertical_folds > 0)
        
        # Calculate fold coverage
        fold_pixels = np.sum(fold_mask)
        total_pixels = gray.shape[0] * gray.shape[1]
        fold_coverage = fold_pixels / total_pixels
        
        # Detect shadow patterns indicating folds
        shadow_score = self._detect_shadow_patterns(gray)
        
        analysis = {
            'coverage': fold_coverage,
            'line_count': len(lines) if lines is not None else 0,
            'shadow_score': shadow_score,
            'severity': 'high' if fold_coverage > self.damage_threshold['folds'] else 
                      'medium' if fold_coverage > self.damage_threshold['folds']/2 else 'low'
        }
        
        return analysis, fold_mask.astype(np.uint8) * 255
    
    def _detect_tears(self, gray: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Detect tears and missing parts in the receipt."""
        # Apply median filter to reduce noise
        filtered = cv2.medianBlur(gray, 5)
        
        # Detect irregular edges that might indicate tears
        edges = cv2.Canny(filtered, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for irregular areas
        tear_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Analyze contour irregularity
        irregular_areas = 0
        total_contour_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Ignore small noise
                # Calculate contour irregularity using convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.7:  # Irregular shape
                        irregular_areas += area
                        cv2.fillPoly(tear_mask, [contour], 255)
                    total_contour_area += area
        
        # Detect sudden brightness changes indicating tears
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find areas with very high gradients (potential tears)
        high_gradient_mask = gradient_magnitude > np.percentile(gradient_magnitude, 98)
        
        # Combine tear detections
        combined_tear_mask = tear_mask | (high_gradient_mask.astype(np.uint8) * 255)
        
        tear_pixels = np.sum(combined_tear_mask > 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        tear_coverage = tear_pixels / total_pixels
        
        analysis = {
            'coverage': tear_coverage,
            'irregular_ratio': irregular_areas / max(total_contour_area, 1),
            'high_gradient_areas': np.sum(high_gradient_mask),
            'severity': 'high' if tear_coverage > self.damage_threshold['tears'] else 
                      'medium' if tear_coverage > self.damage_threshold['tears']/2 else 'low'
        }
        
        return analysis, combined_tear_mask
    
    def _detect_stains(self, image: np.ndarray, hsv: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Detect stains and discoloration in the receipt."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze color variations in HSV space
        _, s, _ = cv2.split(hsv)
        
        # Detect areas with unusual saturation (potential stains)
        saturation_mean = np.mean(s)
        saturation_std = np.std(s)
        stain_mask_sat = s > (saturation_mean + 2 * saturation_std)
        
        # Detect dark spots (potential stains)
        dark_threshold = np.percentile(gray, 15)
        dark_stains = gray < dark_threshold
        
        # Apply morphological operations to clean up stain detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_stains_cleaned = cv2.morphologyEx(dark_stains.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        stain_mask_sat_cleaned = cv2.morphologyEx(stain_mask_sat.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Combine stain detections
        combined_stains = (dark_stains_cleaned > 0) | (stain_mask_sat_cleaned > 0)
        
        # Filter out very small stains (likely noise)
        stain_regions = measure.label(combined_stains)
        filtered_stains = morphology.remove_small_objects(stain_regions > 0, min_size=50)
        
        stain_pixels = np.sum(filtered_stains)
        total_pixels = gray.shape[0] * gray.shape[1]
        stain_coverage = stain_pixels / total_pixels
        
        # Analyze texture uniformity
        texture_score = self._analyze_texture_uniformity(gray)
        
        analysis = {
            'coverage': stain_coverage,
            'texture_score': texture_score,
            'dark_stain_ratio': np.sum(dark_stains_cleaned) / total_pixels,
            'color_variation_ratio': np.sum(stain_mask_sat_cleaned) / total_pixels,
            'severity': 'high' if stain_coverage > self.damage_threshold['stains'] else 
                      'medium' if stain_coverage > self.damage_threshold['stains']/2 else 'low'
        }
        
        return analysis, (filtered_stains.astype(np.uint8) * 255)
    
    def _analyze_contrast(self, gray: np.ndarray) -> Dict:
        """Analyze image contrast and brightness distribution."""
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate contrast using standard deviation
        contrast = np.std(gray)
        
        # Calculate dynamic range
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        
        # Check for over/under exposure
        overexposed = np.sum(gray > 240) / gray.size
        underexposed = np.sum(gray < 15) / gray.size
        
        # Calculate entropy (measure of information content)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'contrast': float(contrast),
            'dynamic_range': float(dynamic_range),
            'overexposed_ratio': float(overexposed),
            'underexposed_ratio': float(underexposed),
            'entropy': float(entropy),
            'quality': 'good' if contrast > 40 and dynamic_range > 150 else 'poor'
        }
    
    def _detect_shadow_patterns(self, gray: np.ndarray) -> float:
        """Detect shadow patterns that indicate folds."""
        # Apply Gaussian blur (for potential future use)
        _ = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Calculate local variations
        local_mean = cv2.blur(gray.astype(np.float32), (30, 30))
        local_variation = np.abs(gray.astype(np.float32) - local_mean)
        
        # Find areas with high local variation (potential shadows)
        shadow_threshold = np.percentile(local_variation, 85)
        shadow_areas = local_variation > shadow_threshold
        
        return np.sum(shadow_areas) / gray.size
    
    def _analyze_texture_uniformity(self, gray: np.ndarray) -> float:
        """Analyze texture uniformity to detect irregular patterns."""
        # Calculate Local Binary Pattern
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Calculate texture uniformity
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-10)
        
        # Calculate uniformity (higher values = more uniform texture)
        uniformity = np.sum(lbp_hist ** 2)
        
        return float(uniformity)
    
    def _calculate_overall_score(self, fold_analysis: Dict, tear_analysis: Dict, 
                                stain_analysis: Dict, contrast_analysis: Dict) -> float:
        """Calculate overall quality score (0-1, higher is better)."""
        # Weight different damage types
        weights = {
            'folds': 0.3,
            'tears': 0.35,
            'stains': 0.25,
            'contrast': 0.1
        }
        
        # Convert damage to quality scores (invert damage ratios)
        fold_score = 1 - min(fold_analysis['coverage'] * 3, 1.0)
        tear_score = 1 - min(tear_analysis['coverage'] * 5, 1.0)
        stain_score = 1 - min(stain_analysis['coverage'] * 3, 1.0)
        contrast_score = min(contrast_analysis['contrast'] / 60, 1.0);
        
        overall_score = (
            fold_score * weights['folds'] +
            tear_score * weights['tears'] +
            stain_score * weights['stains'] +
            contrast_score * weights['contrast']
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _assess_ocr_suitability(self, overall_score: float) -> Dict:
        """Assess OCR suitability based on overall score."""
        if overall_score >= 0.8:
            return {'suitable': True, 'confidence': 'high', 'expected_accuracy': '>95%'}
        elif overall_score >= 0.6:
            return {'suitable': True, 'confidence': 'medium', 'expected_accuracy': '85-95%'}
        elif overall_score >= 0.4:
            return {'suitable': True, 'confidence': 'low', 'expected_accuracy': '70-85%'}
        else:
            return {'suitable': False, 'confidence': 'very_low', 'expected_accuracy': '<70%'}
        
    
    def _create_debug_visualization(self, original_image: np.ndarray, cropped_image: np.ndarray, gray: np.ndarray, 
                              results: Dict, fold_mask: np.ndarray, 
                              tear_mask: np.ndarray, stain_mask: np.ndarray):
        """Create debug visualization showing detected damage with individual damage plots."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Row 1: Original and Cropped images
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Cropped Receipt Image')
        axes[0, 1].axis('off')

        # Grayscale of cropped image
        axes[0, 2].imshow(gray, cmap='gray')
        axes[0, 2].set_title('Grayscale (Cropped)')
        axes[0, 2].axis('off')
        
        # Overall damage overlay
        combined_damage = np.zeros((*gray.shape, 3), dtype=np.uint8)
        combined_damage[fold_mask > 0] = [255, 0, 0]      # Red for folds
        combined_damage[tear_mask > 0] = [0, 255, 0]      # Green for tears  
        combined_damage[stain_mask > 0] = [0, 0, 255]     # Blue for stains
        
        # Overlay on original image
        overlay = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB).copy()
        mask_combined = (fold_mask > 0) | (tear_mask > 0) | (stain_mask > 0)
        overlay[mask_combined] = overlay[mask_combined] * 0.3 + combined_damage[mask_combined] * 0.7
        
        #axes[2, 1].imshow(overlay)
        axes[2, 1].set_title('')
        axes[2, 1].axis('off')

        # Row 2: Individual damage detections
        # Fold detection
        fold_overlay = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB).copy()
        fold_overlay[fold_mask > 0] = fold_overlay[fold_mask > 0] * 0.3 + np.array([255, 0, 0]) * 0.7
        axes[1, 0].imshow(fold_overlay)
        axes[1, 0].set_title(f'Fold Detection\nCoverage: {results["damage_details"]["folds"]["coverage"]:.3f}\nSeverity: {results["damage_details"]["folds"]["severity"]}')
        axes[1, 0].axis('off')
        
        # Tear detection
        tear_overlay = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB).copy()
        tear_overlay[tear_mask > 0] = tear_overlay[tear_mask > 0] * 0.3 + np.array([0, 255, 0]) * 0.7
        axes[1, 1].imshow(tear_overlay)
        axes[1, 1].set_title(f'Tear Detection\nCoverage: {results["damage_details"]["tears"]["coverage"]:.3f}\nSeverity: {results["damage_details"]["tears"]["severity"]}')
        axes[1, 1].axis('off')
        
        # Stain detection
        stain_overlay = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB).copy()
        stain_overlay[stain_mask > 0] = stain_overlay[stain_mask > 0] * 0.3 + np.array([0, 0, 255]) * 0.7
        axes[1, 2].imshow(stain_overlay)
        axes[1, 2].set_title(f'Stain Detection\nCoverage: {results["damage_details"]["stains"]["coverage"]:.3f}\nSeverity: {results["damage_details"]["stains"]["severity"]}')
        axes[1, 2].axis('off')
        
        # Row 3: Analysis summary and recommendations
        # Damage summary
        damage_text = f"""
            Overall Quality Score: {results['overall_score']:.3f}
            OCR Suitable: {results['ocr_suitable']['suitable']}
            Confidence: {results['ocr_suitable']['confidence']}
            Expected Accuracy: {results['ocr_suitable']['expected_accuracy']}

            Detailed Metrics:
            • Fold Coverage: {results['damage_details']['folds']['coverage']:.3f}
            • Tear Coverage: {results['damage_details']['tears']['coverage']:.3f}
            • Stain Coverage: {results['damage_details']['stains']['coverage']:.3f}
            • Contrast Score: {results['damage_details']['contrast']['contrast']:.1f}"""
        
        axes[2, 0].text(0.05, 0.95, damage_text, transform=axes[2, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2, 0].set_title('Damage Analysis Summary')
        axes[2, 0].axis('off')
        
        # Quality score visualization
        score = results['overall_score']
        colors = ['red' if score < 0.4 else 'orange' if score < 0.7 else 'green']
        bars = axes[2, 2].bar(['Overall\nQuality'], [score], color=colors[0], alpha=0.7)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title('Quality Assessment')
        axes[2, 2].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        axes[2, 2].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        axes[2, 2].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Poor (<0.4)')
        axes[2, 2].legend(fontsize=8)
        
        # Add score text on bar
        axes[2, 2].text(0, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Create separate individual damage plots
        self._create_individual_damage_plots(cropped_image, gray, fold_mask, tear_mask, stain_mask, results)
    
    def _create_individual_damage_plots(self, image: np.ndarray, gray: np.ndarray,
                                      fold_mask: np.ndarray, tear_mask: np.ndarray, 
                                      stain_mask: np.ndarray, results: Dict):
        """Create individual plots for each damage type detection."""
        
        # Individual Fold Analysis
        fig_fold, axes_fold = plt.subplots(1, 3, figsize=(15, 5))
        
        axes_fold[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes_fold[0].set_title('Cropped Image')
        axes_fold[0].axis('off')
        
        axes_fold[1].imshow(fold_mask, cmap='hot')
        axes_fold[1].set_title('Fold Detection Mask')
        axes_fold[1].axis('off')
        
        fold_overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        fold_overlay[fold_mask > 0] = fold_overlay[fold_mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6
        axes_fold[2].imshow(fold_overlay)
        axes_fold[2].set_title(f'Fold Overlay\nCoverage: {results["damage_details"]["folds"]["coverage"]:.1%}')
        axes_fold[2].axis('off')
        
        plt.suptitle('FOLD ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Individual Tear Analysis  
        fig_tear, axes_tear = plt.subplots(1, 3, figsize=(15, 5))
        
        axes_tear[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes_tear[0].set_title('Cropped Image')
        axes_tear[0].axis('off')
        
        axes_tear[1].imshow(tear_mask, cmap='hot')
        axes_tear[1].set_title('Tear Detection Mask')
        axes_tear[1].axis('off')
        
        tear_overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        tear_overlay[tear_mask > 0] = tear_overlay[tear_mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6
        axes_tear[2].imshow(tear_overlay)
        axes_tear[2].set_title(f'Tear Overlay\nCoverage: {results["damage_details"]["tears"]["coverage"]:.1%}')
        axes_tear[2].axis('off')
        
        plt.suptitle('TEAR ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Individual Stain Analysis
        fig_stain, axes_stain = plt.subplots(1, 3, figsize=(15, 5))
        
        axes_stain[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes_stain[0].set_title('Cropped Image')
        axes_stain[0].axis('off')
        
        axes_stain[1].imshow(stain_mask, cmap='hot')
        axes_stain[1].set_title('Stain Detection Mask')
        axes_stain[1].axis('off')
        
        stain_overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        stain_overlay[stain_mask > 0] = stain_overlay[stain_mask > 0] * 0.4 + np.array([0, 0, 255]) * 0.6
        axes_stain[2].imshow(stain_overlay)
        axes_stain[2].set_title(f'Stain Overlay\nCoverage: {results["damage_details"]["stains"]["coverage"]:.1%}')
        axes_stain[2].axis('off')
        
        plt.suptitle('STAIN ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Quick test function
def quick_damage_test(image_path: str):
    """Quick test function for single image analysis."""
    analyzer = ReceiptDamageAnalyzer(debug_mode=True)
    return analyzer.analyze_receipt_damage(image_path)

if __name__ == "__main__":
    # Supported image extensions
    
    res = quick_damage_test('dataset1/image_21.jpg')
    print(res)
