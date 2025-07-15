#!/usr/bin/env python3
"""
Test script to verify image quality assessment integration
"""
import sys
import json
from pathlib import Path

# Add current directory to path to import modules
sys.path.append('.')

from image_quality_processor import ImageQualityProcessor
from agno.utils.log import logger

def test_quality_assessment(image_path: str = None):
    """Test quality assessment on a sample image file"""

    # Use provided path or default test image
    if image_path:
        test_image_path = Path(image_path)
    else:
        test_image_path = Path("expense_files/austrian_file.png")

    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return False

    logger.info(f"🧪 ***** Testing quality assessment integration...")
    logger.info(f"Test image: {test_image_path}")
    
    try:
        # Initialize quality processor
        processor = ImageQualityProcessor(document_type='receipt')
        
        # Run quality assessment
        results = processor.assess_image_quality(str(test_image_path))
        
        # Check if assessment was successful
        if 'error' in results:
            logger.error(f"❌ Quality assessment failed: {results['error']}")
            return False
        
        # Display key results
        logger.info(f"✅ ***** Quality assessment successful!")
        logger.info(f"    📊 Overall Score: {results['quality_score']}/100")
        logger.info(f"    🎯 Quality Level: {results['quality_level']}")
        logger.info(f"    ✅ Quality Passed: {results['quality_passed']}")
        logger.info(f"    ⏱️ Processing Time: {results.get('processing_time_seconds', 0):.2f}s")
        
        # Display top issues and recommendations
        if results.get('main_issues'):
            logger.info(f"    ⚠️ Main Issues: {', '.join(results['main_issues'])}")
        
        if results.get('top_recommendations'):
            logger.info(f"    💡 Top Recommendations:")
            for rec in results['top_recommendations']:
                logger.info(f"        • {rec}")
        
        # Save test results to a file (with JSON serialization fix)
        output_filename = f"{test_image_path.stem}_quality_results.json"
        output_file = Path(output_filename)
        with open(output_file, 'w') as f:
            # Convert numpy booleans to Python booleans for JSON serialization
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        logger.info(f"💾 Test results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting quality assessment integration test...")

    # Check if image path provided as command line argument
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        logger.info(f"📁 Using provided image: {image_path}")
    else:
        logger.info("📁 Using default test image")

    success = test_quality_assessment(image_path)

    if success:
        logger.info("✅ ***** Integration test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ ***** Integration test FAILED!")
        sys.exit(1)