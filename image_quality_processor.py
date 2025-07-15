import os
import pathlib
from typing import Dict, List, Optional
from agno.utils.log import logger

from quality_assessment.quality_assessor import QualityAssessor
from quality_assessment.utils import ImageLoadError

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class ImageQualityProcessor:
    """
    Wrapper class for image quality assessment in the expense processing system
    """
    
    def __init__(self, document_type: str = 'receipt'):
        """
        Initialize the image quality processor
        
        Args:
            document_type: Type of document to assess ('receipt', 'a4', 'letter', 'id_card', 'default')
        """
        self.document_type = document_type
        self.quality_assessor = None
        logger.info(f"🎯 ***** Initializing Image Quality Processor for {document_type} documents...")
    
    def _ensure_assessor_initialized(self):
        """Lazy initialization of quality assessor to avoid unnecessary imports"""
        if self.quality_assessor is None:
            self.quality_assessor = QualityAssessor(self.document_type)
    
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
        Perform quality assessment on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing quality assessment results
        """
        logger.info(f"🔍 ***** Starting quality assessment for: {os.path.basename(image_path)}")
        
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
            results = self.quality_assessor.assess_image(image_path)
            
            # Add convenience fields for easier integration
            if 'error' not in results:
                results['quality_passed'] = results['overall_assessment']['pass_fail']
                results['quality_score'] = results['overall_assessment']['score']
                results['quality_level'] = results['overall_assessment']['level']
                results['main_issues'] = results['overall_assessment']['issues_summary'][:3]
                results['top_recommendations'] = results['overall_assessment']['recommendations'][:3]
                
                logger.info(f"✅ Quality assessment complete - Score: {results['quality_score']}/100 ({results['quality_level']})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed for {image_path}: {str(e)}")
            return {
                'error': f'Quality assessment failed: {str(e)}',
                'image_path': image_path
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


def assess_image_quality_for_expense(image_path: str, document_type: str = 'receipt') -> Dict:
    """
    Convenience function for single image quality assessment
    
    Args:
        image_path: Path to the image file
        document_type: Type of document (default: 'receipt')
        
    Returns:
        Quality assessment results
    """
    processor = ImageQualityProcessor(document_type)
    return processor.assess_image_quality(image_path)


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