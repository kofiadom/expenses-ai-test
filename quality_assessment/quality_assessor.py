import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

from .utils import load_image, ImageLoadError, detect_image_type
from .quality_modules.resolution_checker import ResolutionChecker
from .quality_modules.blur_detector import BlurDetector
from .quality_modules.glare_analyzer import GlareAnalyzer
from .quality_modules.completeness_checker import CompletenessChecker
from .quality_modules.damage_detector import DamageDetector
from agno.utils.log import logger


class QualityAssessor:
    
    def __init__(self, document_type: str = 'receipt'):
        """
        Initialize quality assessor
        
        Args:
            document_type: Type of document ('receipt', 'a4', 'letter', 'id_card', 'default')
        """
        self.document_type = document_type
        logger.info(f"🚀 ***** Initializing Quality Assessor for {document_type} documents...")
        
        self.resolution_checker = ResolutionChecker(document_type)
        self.blur_detector = BlurDetector()
        self.glare_analyzer = GlareAnalyzer()
        self.completeness_checker = CompletenessChecker()
        self.damage_detector = DamageDetector()
        
        self.weights = {
            'resolution': 0.20,
            'blur': 0.25,
            'glare': 0.20,
            'completeness': 0.20,
            'damage': 0.15
        }
        
        logger.info(f"✅ ***** Quality Assessor initialized.")
    
    def assess_image(self, image_path: str) -> Dict:
        """
        Perform comprehensive quality assessment on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all assessment results
        """
        logger.info(f"📸 ***** Starting quality assessment for: {image_path}")
        start_time = datetime.now()
        
        try:
            cv2_image, pil_image = load_image(image_path)
            
            image_type_info = detect_image_type(cv2_image, pil_image)
            is_screenshot = image_type_info['is_digital_screenshot']
            
            logger.info(f"🔬 Running quality checks for {image_type_info['image_subtype']}...")
            
            resolution_results = self.resolution_checker.assess_resolution(cv2_image, pil_image)
            blur_results = self.blur_detector.assess_blur(cv2_image)
            
            # Adjust resolution scoring for screenshots
            if is_screenshot:
                width, height = resolution_results['dimensions']['width'], resolution_results['dimensions']['height']
                min_dimension = min(width, height)
                
                if min_dimension >= 1080:
                    resolution_score = 100.0
                elif min_dimension >= 720:
                    resolution_score = 80.0
                elif min_dimension >= 480:
                    resolution_score = 60.0
                else:
                    resolution_score = 40.0
                
                resolution_results['quality']['score'] = resolution_score
                resolution_results['quality']['level'] = 'Digital Quality'
                logger.info(f"📱 Adjusted resolution score for screenshot: {resolution_score}")
            
            # Adjust glare analysis for screenshots
            if is_screenshot:
                glare_results = self.glare_analyzer.assess_glare(cv2_image)
                
                # If it's mostly white background (common for screenshots), adjust score
                if glare_results['glare_analysis']['glare_coverage_percent'] > 70:
                    glare_results['glare_analysis']['glare_score'] = 95.0
                    glare_results['glare_analysis']['glare_level'] = 'None (Digital)'
                    glare_results['recommendations'] = ['✅ Digital image with clean background.']
                    logger.info(f"📱 Adjusted glare score for screenshot with white background")
            else:
                glare_results = self.glare_analyzer.assess_glare(cv2_image)
            
            # Skip completeness check for screenshots
            if is_screenshot:
                completeness_results = {
                    'boundary_detected': True,
                    'completeness_score': 100.0,
                    'completeness_level': 'Digital Document',
                    'edge_analysis': {
                        'edge_coverage': 100.0,
                        'has_gaps': False,
                        'num_gaps': 0
                    },
                    'corner_analysis': {
                        'visible_corners': 4,
                        'missing_corners': [],
                        'is_rectangular': True
                    },
                    'issues': [],
                    'boundary_points': [],
                    'recommendations': ['✅ Digital screenshot - no physical boundaries to check.']
                }
                logger.info(f"📱 Skipped completeness check for screenshot")
            else:
                completeness_results = self.completeness_checker.assess_completeness(cv2_image)
            
            # Skip damage detection for screenshots
            if is_screenshot:
                damage_results = {
                    'damage_score': 100.0,
                    'damage_level': 'Digital (No Physical Damage)',
                    'damage_types': [],
                    'stain_analysis': {
                        'count': 0,
                        'coverage_percent': 0.0,
                        'regions': []
                    },
                    'tear_analysis': {
                        'count': 0,
                        'max_length': 0,
                        'regions': []
                    },
                    'fold_analysis': {
                        'count': 0,
                        'pattern': 'none',
                        'lines': []
                    },
                    'recommendations': ['✅ Digital image - no physical damage possible.']
                }
                logger.info(f"📱 Skipped damage detection for screenshot")
            else:
                damage_results = self.damage_detector.assess_damage(cv2_image)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score({
                'resolution': resolution_results['quality']['score'],
                'blur': blur_results['metrics']['blur_score'],
                'glare': glare_results['glare_analysis']['glare_score'],
                'completeness': completeness_results['completeness_score'],
                'damage': damage_results['damage_score']
            })
            
            overall_level = self._determine_quality_level(overall_score)
            
            all_issues = self._compile_issues(
                resolution_results, blur_results, glare_results, 
                completeness_results, damage_results
            )
            
            consolidated_recommendations = self._consolidate_recommendations(
                resolution_results, blur_results, glare_results,
                completeness_results, damage_results
            )
            
            if is_screenshot:
                consolidated_recommendations.insert(0, 
                    f"📱 This is a {image_type_info['image_subtype']} - physical document checks have been adjusted.")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'image_path': image_path,
                'document_type': self.document_type,
                'image_type_detection': image_type_info,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'overall_assessment': {
                    'score': round(overall_score, 1),
                    'level': overall_level,
                    'pass_fail': overall_score >= 70,  # 70 is passing threshold
                    'issues_summary': all_issues,
                    'recommendations': consolidated_recommendations[:5]  # Top 5 recommendations
                },
                'detailed_results': {
                    'resolution': resolution_results,
                    'blur': blur_results,
                    'glare': glare_results,
                    'completeness': completeness_results,
                    'damage': damage_results
                },
                'score_breakdown': {
                    'resolution': {
                        'score': resolution_results['quality']['score'],
                        'weight': self.weights['resolution'],
                        'contribution': resolution_results['quality']['score'] * self.weights['resolution']
                    },
                    'blur': {
                        'score': blur_results['metrics']['blur_score'],
                        'weight': self.weights['blur'],
                        'contribution': blur_results['metrics']['blur_score'] * self.weights['blur']
                    },
                    'glare': {
                        'score': glare_results['glare_analysis']['glare_score'],
                        'weight': self.weights['glare'],
                        'contribution': glare_results['glare_analysis']['glare_score'] * self.weights['glare']
                    },
                    'completeness': {
                        'score': completeness_results['completeness_score'],
                        'weight': self.weights['completeness'],
                        'contribution': completeness_results['completeness_score'] * self.weights['completeness']
                    },
                    'damage': {
                        'score': damage_results['damage_score'],
                        'weight': self.weights['damage'],
                        'contribution': damage_results['damage_score'] * self.weights['damage']
                    }
                }
            }
            
            logger.info(f"✅ ***** Quality assessment complete. Overall score: {overall_score}/100")
            return results
            
        except ImageLoadError as e:
            logger.error(f"❌ Failed to load image: {str(e)}")
            return {
                'error': str(e),
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error during assessment: {str(e)}")
            return {
                'error': f"Assessment failed: {str(e)}",
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total = sum(scores[metric] * self.weights[metric] for metric in scores)
        return total
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score"""
        if score >= 95:
            return 'Excellent'
        elif score >= 85:
            return 'Good'
        elif score >= 70:
            return 'Acceptable'
        elif score >= 50:
            return 'Poor'
        else:
            return 'Unacceptable'
    
    def _compile_issues(self, *results) -> List[str]:
        """Compile all issues from different assessment modules"""
        issues = []
        
        for result in results:
            if 'issues' in result:
                issues.extend(result['issues'])
            elif 'quality' in result and 'level' in result['quality']:
                if result['quality']['level'] in ['Poor', 'Fair']:
                    issues.append(f"{result['quality']['level']} resolution quality")
            elif 'metrics' in result and 'is_blurry' in result['metrics']:
                if result['metrics']['is_blurry']:
                    issues.append(f"Image blur detected ({result['metrics']['blur_level']})")
            elif 'glare_analysis' in result:
                if result['glare_analysis']['glare_level'] in ['Significant', 'Severe']:
                    issues.append(f"{result['glare_analysis']['glare_level']} glare detected")
            elif 'damage_types' in result:
                if result['damage_types']:
                    issues.append(f"Physical damage: {', '.join(result['damage_types'])}")
        
        return issues
    
    def _consolidate_recommendations(self, *results) -> List[str]:
        """Consolidate and prioritize recommendations from all modules"""
        all_recommendations = []
        
        for result in results:
            if 'recommendations' in result:
                all_recommendations.extend(result['recommendations'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        # Prioritize recommendations: critical first, then improvements, then others
        critical = [r for r in unique_recommendations if '⚠️' in r or '❌' in r]
        improvements = [r for r in unique_recommendations if '📈' in r or '💡' in r]
        others = [r for r in unique_recommendations if r not in critical and r not in improvements]
        
        return critical + improvements + others
    
    def save_results_json(self, results: Dict, output_path: str) -> None:
        """Save results to JSON file with proper type conversion"""
        def convert_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if obj is None:
                return None
            
            # Handle numpy scalars
            if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
                try:
                    return obj.item()
                except:
                    pass
            
            # Handle specific numpy types
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_types(item) for item in obj]
            else:
                # Handle any other numpy types
                if hasattr(obj, '__module__') and obj.__module__ and 'numpy' in obj.__module__:
                    try:
                        if hasattr(obj, 'tolist'):
                            return obj.tolist()
                        elif hasattr(obj, '__float__'):
                            return float(obj)
                        elif hasattr(obj, '__int__'):
                            return int(obj)
                        else:
                            return str(obj)
                    except:
                        return str(obj)
                return obj
        
        try:
            logger.info(f"🔄 Starting JSON conversion for results...")
            json_safe_results = convert_types(results)
            
            with open(output_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            logger.info(f"💾 Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            raise 