#!/usr/bin/env python3
"""
Test script to verify image quality assessment integration
"""
import sys
import json
import glob
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

def test_batch_quality_assessment(directory_path: str = "expense_files"):
    """Test quality assessment on all image files in a directory"""

    # Supported image extensions
    supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp', '*.gif']

    # Find all image files in the directory
    image_files = []
    directory = Path(directory_path)

    if not directory.exists():
        logger.error(f"Directory not found: {directory_path}")
        return False

    for extension in supported_extensions:
        image_files.extend(directory.glob(extension))
        image_files.extend(directory.glob(extension.upper()))  # Also check uppercase

    if not image_files:
        logger.error(f"No image files found in directory: {directory_path}")
        return False

    logger.info(f"🧪 ***** Testing batch quality assessment...")
    logger.info(f"Directory: {directory_path}")
    logger.info(f"Found {len(image_files)} image files")

    try:
        # Initialize quality processor
        quality_processor = ImageQualityProcessor(document_type='receipt')

        # Create quality_reports directory (same as main workflow)
        quality_reports_dir = Path("quality_reports")
        quality_reports_dir.mkdir(exist_ok=True)

        # Process all images
        successful_count = 0
        failed_count = 0
        saved_files = []

        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📸 Processing {i}/{len(image_files)}: {image_path.name}")

            try:
                # Run quality assessment
                results = quality_processor.assess_image_quality(str(image_path))

                if 'error' in results:
                    logger.error(f"❌ Quality assessment failed for {image_path.name}: {results['error']}")
                    failed_count += 1
                else:
                    # Log key metrics
                    overall_score = results.get('overall_assessment', {}).get('score', 0)
                    quality_level = results.get('overall_assessment', {}).get('level', 'Unknown')
                    passed = results.get('overall_assessment', {}).get('pass_fail', False)

                    logger.info(f"✅ {image_path.name}: Score {overall_score:.1f}/100, Level: {quality_level}, {'PASS' if passed else 'FAIL'}")

                    # Save individual quality result file (same format as main workflow)
                    quality_filename = f"{image_path.stem}_quality.json"
                    quality_file_path = quality_reports_dir / quality_filename

                    with open(quality_file_path, 'w') as f:
                        # Convert numpy booleans to Python booleans for JSON serialization
                        serializable_results = json.loads(json.dumps(results, default=str))
                        json.dump(serializable_results, f, indent=2)

                    saved_files.append(quality_file_path)
                    successful_count += 1

            except Exception as e:
                logger.error(f"❌ Exception processing {image_path.name}: {str(e)}")
                failed_count += 1

        # Log saved files
        if saved_files:
            logger.info(f"💾 Individual quality reports saved to quality_reports/ directory:")
            for file_path in saved_files:
                logger.info(f"   • {file_path.name}")

        # Summary
        logger.info(f"📊 ***** Batch processing summary:")
        logger.info(f"✅ Successful: {successful_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"📁 Total files: {len(image_files)}")

        return successful_count > 0

    except Exception as e:
        logger.error(f"❌ Batch test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting quality assessment integration test...")

    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check for batch processing flag
        if arg == "--batch" or arg == "-b":
            # Batch processing mode
            directory = sys.argv[2] if len(sys.argv) > 2 else "expense_files"
            logger.info(f"📁 Batch processing mode - Directory: {directory}")
            success = test_batch_quality_assessment(directory)
        elif arg == "--help" or arg == "-h":
            # Help message
            print("""
🧪 Quality Assessment Integration Test

Usage:
  python test_quality_integration.py                    # Test default image
  python test_quality_integration.py <image_path>       # Test specific image
  python test_quality_integration.py --batch [dir]      # Test all images in directory
  python test_quality_integration.py -b [dir]           # Short form for batch
  python test_quality_integration.py --help             # Show this help

Examples:
  python test_quality_integration.py
  python test_quality_integration.py my_image.jpg
  python test_quality_integration.py --batch
  python test_quality_integration.py --batch my_images/
  python test_quality_integration.py -b expense_files/

Batch mode processes all image files (PNG, JPG, JPEG, TIFF, BMP, GIF) in the specified directory.
Default directory for batch mode is 'expense_files/'.
            """)
            sys.exit(0)
        else:
            # Single image mode
            logger.info(f"📁 Single image mode - File: {arg}")
            success = test_quality_assessment(arg)
    else:
        # Default mode - single image
        logger.info("📁 Default mode - Using default test image")
        success = test_quality_assessment()

    if success:
        logger.info("✅ ***** Integration test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ ***** Integration test FAILED!")
        sys.exit(1)