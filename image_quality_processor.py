import os
import pathlib
from typing import Dict, List, Optional
from agno.utils.log import logger

from quality_assessment.quality_assessor import QualityAssessor
from llm_image_quality_assessor import LLMImageQualityAssessor
from quality_assessment.utils import ImageLoadError

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class ImageQualityProcessor:
    """
    Wrapper class for image quality assessment in the expense processing system
    Supports both OpenCV-based and LLM-based assessment methods
    """

    def __init__(self, document_type: str = 'receipt', assessment_method: str = 'opencv'):
        """
        Initialize the image quality processor

        Args:
            document_type: Type of document to assess ('receipt', 'a4', 'letter', 'id_card', 'default')
            assessment_method: Assessment method to use ('opencv', 'llm', 'both')
        """
        self.document_type = document_type
        self.assessment_method = assessment_method.lower()
        self.quality_assessor = None
        self.llm_assessor = None

        if self.assessment_method not in ['opencv', 'llm', 'both']:
            raise ValueError("assessment_method must be 'opencv', 'llm', or 'both'")

        logger.info(f"🎯 ***** Initializing Image Quality Processor for {document_type} documents using {assessment_method} method...")
    
    def _ensure_assessor_initialized(self):
        """Lazy initialization of quality assessors to avoid unnecessary imports"""
        if self.assessment_method in ['opencv', 'both'] and self.quality_assessor is None:
            self.quality_assessor = QualityAssessor(self.document_type)

        if self.assessment_method in ['llm', 'both'] and self.llm_assessor is None:
            try:
                self.llm_assessor = LLMImageQualityAssessor()
                logger.info("✅ LLM assessor initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM assessor: {str(e)}")
                if self.assessment_method == 'llm':
                    raise
                else:
                    logger.warning("⚠️ Falling back to OpenCV-only assessment")
    
    def is_image_file(self, file_path: str) -> bool:
        """
        Check if a file is a supported image format
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a supported image format
        """
        file_extension = pathlib.Path(file_path).suffix.lower()
        return file_extension in SUPPORTED_IMAGE_EXTENSIONS
    
    def assess_image_quality(self, image_path: str) -> Dict:
        """
        Perform quality assessment on a single image using the configured method

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing quality assessment results
        """
        logger.info(f"🔍 ***** Starting {self.assessment_method} quality assessment for: {os.path.basename(image_path)}")

        if not self.is_image_file(image_path):
            logger.warning(f"⚠️ File is not a supported image format: {image_path}")
            return {
                'error': 'Unsupported file format',
                'image_path': image_path,
                'supported_formats': list(SUPPORTED_IMAGE_EXTENSIONS)
            }

        if not os.path.exists(image_path):
            logger.error(f"❌ Image file not found: {image_path}")
            return {
                'error': 'Image file not found',
                'image_path': image_path
            }

        try:
            self._ensure_assessor_initialized()

            if self.assessment_method == 'opencv':
                return self._assess_with_opencv(image_path)
            elif self.assessment_method == 'llm':
                return self._assess_with_llm(image_path)
            elif self.assessment_method == 'both':
                return self._assess_with_both(image_path)

        except Exception as e:
            logger.error(f"❌ Quality assessment failed for {image_path}: {str(e)}")
            return {
                'error': f'Quality assessment failed: {str(e)}',
                'image_path': image_path
            }

    def _assess_with_opencv(self, image_path: str) -> Dict:
        """Perform OpenCV-based quality assessment"""
        results = self.quality_assessor.assess_image(image_path)

        # Add convenience fields for easier integration
        if 'error' not in results:
            results['assessment_method'] = 'opencv'
            results['quality_passed'] = results['overall_assessment']['pass_fail']
            results['quality_score'] = results['overall_assessment']['score']
            results['quality_level'] = results['overall_assessment']['level']
            results['main_issues'] = results['overall_assessment']['issues_summary'][:3]
            results['top_recommendations'] = results['overall_assessment']['recommendations'][:3]

            logger.info(f"✅ OpenCV quality assessment complete - Score: {results['quality_score']}/100 ({results['quality_level']})")

        return results

    def _assess_with_llm(self, image_path: str) -> Dict:
        """Perform LLM-based quality assessment"""
        if not self.llm_assessor:
            raise Exception("LLM assessor not available")

        assessment = self.llm_assessor.assess_image_quality_sync(image_path)
        results = self.llm_assessor.format_assessment_for_workflow(assessment, image_path)

        logger.info(f"✅ LLM quality assessment complete - Score: {results['quality_score']}/100 ({results['quality_level']})")
        return results

    def _assess_with_both(self, image_path: str) -> Dict:
        """Perform both OpenCV and LLM assessments and combine results"""
        opencv_results = self._assess_with_opencv(image_path)

        try:
            llm_results = self._assess_with_llm(image_path)
        except Exception as e:
            logger.warning(f"⚠️ LLM assessment failed, using OpenCV only: {str(e)}")
            opencv_results['assessment_method'] = 'opencv_only_fallback'
            return opencv_results

        # Combine results
        combined_results = {
            'image_path': image_path,
            'assessment_method': 'both',
            'opencv_results': opencv_results,
            'llm_results': llm_results,
            'comparison': self._compare_assessments(opencv_results, llm_results)
        }

        # Use LLM results as primary for convenience fields
        combined_results.update({
            'quality_passed': llm_results['quality_passed'],
            'quality_score': llm_results['quality_score'],
            'quality_level': llm_results['quality_level'],
            'main_issues': llm_results['main_issues'],
            'top_recommendations': llm_results['top_recommendations']
        })

        logger.info(f"✅ Combined assessment complete - OpenCV: {opencv_results['quality_score']}/100, "
                   f"LLM: {llm_results['quality_score']}/100")

        return combined_results

    def _compare_assessments(self, opencv_results: Dict, llm_results: Dict) -> Dict:
        """Compare OpenCV and LLM assessment results"""
        opencv_score = opencv_results.get('quality_score', 0)
        llm_score = llm_results.get('quality_score', 0)
        score_difference = abs(opencv_score - llm_score)

        opencv_passed = opencv_results.get('quality_passed', False)
        llm_passed = llm_results.get('quality_passed', False)
        agreement = opencv_passed == llm_passed

        return {
            'score_difference': score_difference,
            'pass_fail_agreement': agreement,
            'opencv_score': opencv_score,
            'llm_score': llm_score,
            'opencv_passed': opencv_passed,
            'llm_passed': llm_passed,
            'assessment_consensus': 'high' if score_difference < 20 and agreement else 'low'
        }
    
    def assess_multiple_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Perform quality assessment on multiple images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of quality assessment results
        """
        logger.info(f"📊 ***** Starting quality assessment for {len(image_paths)} images...")
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"📸 Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.assess_image_quality(image_path)
            results.append(result)
        
        # Generate summary statistics
        successful_assessments = [r for r in results if 'error' not in r]
        if successful_assessments:
            average_score = sum(r['quality_score'] for r in successful_assessments) / len(successful_assessments)
            passing_count = sum(1 for r in successful_assessments if r['quality_passed'])
            
            logger.info(f"✅ ***** Batch assessment complete:")
            logger.info(f"    📊 Processed: {len(results)} images")
            logger.info(f"    ✅ Successful: {len(successful_assessments)}")
            logger.info(f"    🎯 Average score: {average_score:.1f}/100")
            logger.info(f"    📈 Passing quality: {passing_count}/{len(successful_assessments)}")
        
        return results
    
    def filter_image_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter a list of file paths to only include supported image files
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            List of image file paths
        """
        image_files = [fp for fp in file_paths if self.is_image_file(fp)]
        
        if len(image_files) != len(file_paths):
            non_image_count = len(file_paths) - len(image_files)
            logger.info(f"📁 Filtered files: {len(image_files)} images, {non_image_count} non-images")
        
        return image_files
    
    def should_process_quality(self, file_path: str) -> bool:
        """
        Determine if a file should go through quality assessment
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be quality assessed
        """
        return self.is_image_file(file_path) and os.path.exists(file_path)
    
    def get_quality_summary(self, results: List[Dict]) -> Dict:
        """
        Generate a summary of quality assessment results
        
        Args:
            results: List of quality assessment results
            
        Returns:
            Summary statistics dictionary
        """
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {
                'total_processed': len(results),
                'successful_assessments': 0,
                'average_score': 0,
                'passing_count': 0,
                'passing_percentage': 0,
                'quality_distribution': {}
            }
        
        scores = [r['quality_score'] for r in successful_results]
        passing_results = [r for r in successful_results if r['quality_passed']]
        
        # Quality level distribution
        quality_levels = {}
        for result in successful_results:
            level = result['quality_level']
            quality_levels[level] = quality_levels.get(level, 0) + 1
        
        summary = {
            'total_processed': len(results),
            'successful_assessments': len(successful_results),
            'average_score': round(sum(scores) / len(scores), 1),
            'min_score': min(scores),
            'max_score': max(scores),
            'passing_count': len(passing_results),
            'passing_percentage': round(len(passing_results) / len(successful_results) * 100, 1),
            'quality_distribution': quality_levels
        }
        
        return summary


def assess_image_quality_for_expense(image_path: str, document_type: str = 'receipt', assessment_method: str = 'opencv') -> Dict:
    """
    Convenience function for single image quality assessment

    Args:
        image_path: Path to the image file
        document_type: Type of document (default: 'receipt')
        assessment_method: Assessment method to use ('opencv', 'llm', 'both')

    Returns:
        Quality assessment results
    """
    processor = ImageQualityProcessor(document_type, assessment_method)
    return processor.assess_image_quality(image_path)


def assess_image_quality_llm(image_path: str, document_type: str = 'receipt') -> Dict:
    """
    Convenience function for LLM-based image quality assessment

    Args:
        image_path: Path to the image file
        document_type: Type of document (default: 'receipt')

    Returns:
        LLM quality assessment results
    """
    return assess_image_quality_for_expense(image_path, document_type, 'llm')


def assess_image_quality_combined(image_path: str, document_type: str = 'receipt') -> Dict:
    """
    Convenience function for combined OpenCV + LLM image quality assessment

    Args:
        image_path: Path to the image file
        document_type: Type of document (default: 'receipt')

    Returns:
        Combined quality assessment results
    """
    return assess_image_quality_for_expense(image_path, document_type, 'both')


def filter_images_from_files(file_paths: List[str]) -> List[str]:
    """
    Convenience function to filter image files from a list of file paths
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of image file paths
    """
    processor = ImageQualityProcessor()
    return processor.filter_image_files(file_paths) 