import requests
import pathlib
import time
import json
from typing import Dict, List, Tuple
from agno.utils.log import logger

from image_quality_processor import ImageQualityProcessor, SUPPORTED_IMAGE_EXTENSIONS
from llm_image_quality_assessor import LLMImageQualityAssessor
from llm_quality_validator import ImageQualityUQLMValidator
from langchain_anthropic import ChatAnthropic
import asyncio
import os

SUPPORTED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

class LlamaIndexAPI:
    """
    A minimal class to interact with the LlamaIndex API for file parsing.
    """
    BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
        }

    def upload_file(self, file_path: pathlib.Path, parse_mode="parse_page_with_lvm", vendor_multimodal_model_name="anthropic-sonnet-3.7") -> dict:
        url = f"{self.BASE_URL}/upload"
        files = {
            "file": (
                file_path.name,
                open(file_path, "rb"),
                "application/pdf",
            ),
        }
        data = {
            "parse_mode": parse_mode,
            "vendor_multimodal_model_name": vendor_multimodal_model_name,
            "input_url": "",
            "structured_output": False,
            "disable_ocr": False,
            "disable_image_extraction": False,
            "adaptive_long_table": False,
            "annotate_links": False,
            "do_not_unroll_columns": False,
            "html_make_all_elements_visible": False,
            "html_remove_navigation_elements": False,
            "html_remove_fixed_elements": False,
            "guess_xlsx_sheet_name": False,
            "do_not_cache": False,
            "invalidate_cache": False,
            "output_pdf_of_document": False,
            "take_screenshot": False,
            "is_formatting_instruction": True,
        }
        try:
            response = requests.post(url, headers=self.headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"LlamaParse Upload Error: {str(e)} for file {file_path}")
            return {"error": str(e)}
        finally:
            files["file"][1].close()

    def get_job_status(self, job_id: str) -> dict:
        url = f"{self.BASE_URL}/job/{job_id}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"LlamaParse Status Check Error: {str(e)} for job {job_id}")
            return {"error": str(e)}

    def get_job_result_markdown(self, job_id: str) -> dict:
        url = f"{self.BASE_URL}/job/{job_id}/result/markdown"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"LlamaParse Result Error: {str(e)} for job {job_id}")
            return {"error": str(e)}

def assess_image_quality_before_extraction(file_path: pathlib.Path, quality_processor: ImageQualityProcessor, quality_output_dir: pathlib.Path) -> Dict:
    """
    Assess image quality before LlamaParse extraction and save results (OpenCV method)

    Args:
        file_path: Path to the image file
        quality_processor: ImageQualityProcessor instance
        quality_output_dir: Directory to save quality assessment results

    Returns:
        Quality assessment results dictionary
    """
    logger.info(f"🔍 Running quality assessment for {file_path.name}...")
    
    try:
        # Perform quality assessment
        quality_results = quality_processor.assess_image_quality(str(file_path))
        
        # Save quality results to JSON file
        quality_filename = f"{file_path.stem}_quality.json"
        quality_file_path = quality_output_dir / quality_filename

        with open(quality_file_path, 'w') as f:
            # Fix JSON serialization for numpy booleans and other non-serializable types
            serializable_results = json.loads(json.dumps(quality_results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        if 'error' not in quality_results:
            score = quality_results['quality_score']
            level = quality_results['quality_level'] 
            passed = quality_results['quality_passed']
            
            logger.info(f"✅ Quality assessment saved: {quality_filename}")
            logger.info(f"    📊 Score: {score}/100 ({level})")
            logger.info(f"    🎯 Status: {'PASS' if passed else 'FAIL'}")
            
            if not passed:
                logger.warning(f"⚠️ Image quality below threshold, but processing will continue")
        else:
            logger.error(f"❌ Quality assessment failed: {quality_results.get('error', 'Unknown error')}")
        
        return quality_results
        
    except Exception as e:
        logger.error(f"❌ Quality assessment exception for {file_path.name}: {str(e)}")
        return {
            'error': f'Quality assessment failed: {str(e)}',
            'image_path': str(file_path)
        }


def assess_image_quality_llm_separate(file_path: pathlib.Path, llm_quality_output_dir: pathlib.Path,
                                     enable_validation: bool = True, opencv_assessment: Dict = None) -> Dict:
    """
    Perform separate LLM-based image quality assessment with optional UQLM validation and save results

    Args:
        file_path: Path to the image file
        llm_quality_output_dir: Directory to save LLM quality assessment results
        enable_validation: Whether to perform UQLM validation of the LLM assessment
        opencv_assessment: Optional OpenCV assessment for validation comparison

    Returns:
        LLM quality assessment results dictionary (with validation if enabled)
    """
    try:
        logger.info(f"🤖 Starting LLM quality assessment for: {file_path.name}")
        assessment_start_time = time.time()

        # Initialize LLM assessor
        llm_assessor = LLMImageQualityAssessor()

        # Perform LLM assessment
        assessment = llm_assessor.assess_image_quality_sync(str(file_path))

        # Format results for workflow
        llm_result = llm_assessor.format_assessment_for_workflow(assessment, str(file_path))

        # Save LLM assessment results to separate directory
        llm_quality_output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"llm_quality_{file_path.stem}.json"
        output_path = llm_quality_output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_result, f, indent=2, ensure_ascii=False)

        assessment_time = time.time() - assessment_start_time
        logger.info(f"✅ LLM quality assessment saved to: {output_path} (Time: {assessment_time:.2f}s)")
        logger.info(f"    🤖 LLM Score: {llm_result['overall_quality_score']}/10")
        logger.info(f"    🎯 LLM Status: {'PASS' if llm_result['suitable_for_extraction'] else 'FAIL'}")

        # Count detected issues
        detected_issues = []
        for key, value in llm_result.items():
            if isinstance(value, dict) and value.get('detected', False):
                issue_name = key.replace('_', ' ').title()
                detected_issues.append(issue_name)

        if detected_issues:
            logger.info(f"    🔍 Detected Issues: {', '.join(detected_issues[:3])}")

        # Perform UQLM validation if enabled
        validation_result = None
        if enable_validation:
            try:
                logger.info(f"🎯 Starting UQLM validation for LLM quality assessment: {file_path.name}")
                validation_start_time = time.time()

                # Initialize validator with primary LLM
                primary_llm = ChatAnthropic(model="claude-3-7-sonnet-20250219",
                                          api_key=os.getenv("ANTHROPIC_API_KEY"))
                validator = ImageQualityUQLMValidator(primary_llm, logger)

                # Run validation asynchronously
                validation_result = asyncio.run(validator.validate_quality_assessment(
                    llm_assessment=llm_result,
                    image_path=str(file_path),
                    opencv_assessment=opencv_assessment
                ))

                # Save validation results to separate directory
                validation_dir = pathlib.Path("llm_quality_validation_results")
                validation_dir.mkdir(parents=True, exist_ok=True)
                validation_filename = f"llm_quality_validation_{file_path.stem}.json"
                validation_path = validation_dir / validation_filename

                with open(validation_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_result, f, indent=2, ensure_ascii=False)

                validation_time = time.time() - validation_start_time
                logger.info(f"✅ UQLM validation saved to: {validation_path} (Time: {validation_time:.2f}s)")
                logger.info(f"    🎯 Validation Confidence: {validation_result['validation_summary']['overall_confidence']:.2f}")
                logger.info(f"    📊 Reliability Level: {validation_result['validation_summary']['reliability_level']}")

                # Add validation summary to the result
                llm_result['uqlm_validation'] = validation_result

            except Exception as validation_error:
                logger.warning(f"⚠️ UQLM validation failed for {file_path.name}: {str(validation_error)}")
                logger.info("📋 Continuing without validation results")

        return llm_result

    except Exception as e:
        logger.error(f"❌ LLM quality assessment failed for {file_path.name}: {str(e)}")
        return {
            'error': f'LLM quality assessment failed: {str(e)}',
            'image_path': str(file_path),
            'assessment_method': 'llm'
        }


def extract_markdown(file_path: pathlib.Path, api: LlamaIndexAPI, output_dir: pathlib.Path):
    """Extract markdown from file using LlamaParse (unchanged from original)"""
    logger.info(f"🔄 Uploading {file_path.name} to LlamaParse...")
    upload_response = api.upload_file(file_path)
    if 'error' in upload_response:
        logger.error(f"Error uploading {file_path}: {upload_response['error']}")
        return
    job_id = upload_response.get('id') or upload_response.get('job_id')
    if not job_id:
        logger.error(f"No job_id returned for {file_path}. Response: {upload_response}")
        return
    logger.info(f"Job ID for {file_path}: {job_id}. Waiting for completion...")
    for _ in range(60):  # Wait up to 5 minutes
        status_response = api.get_job_status(job_id)
        status = status_response.get('status')
        logger.debug(f"Polled status for {file_path}: {status}")
        if status and status.lower() in ('completed', 'success'):
            break
        elif status and status.lower() == 'failed':
            logger.error(f"Job failed for {file_path}: {status_response}")
            return
        time.sleep(5)
    else:
        logger.error(f"Timeout waiting for job {job_id} for {file_path}")
        return
    logger.debug(f"Job completed for {file_path}, fetching result...")
    result_response = api.get_job_result_markdown(job_id)
    if 'error' in result_response:
        logger.error(f"Error fetching result for {file_path}: {result_response['error']}")
        return
    markdown = result_response.get('markdown') or result_response.get('result')
    if not markdown:
        logger.error(f"No markdown found in result for {file_path}. Response: {result_response}")
        return
    output_file = output_dir / (file_path.stem + '.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    logger.info(f"✅ Successfully saved markdown to {output_file}")

def process_expense_files(api_key: str, input_folder: str = "expense_files", output_dir: str = "llamaparse_output") -> list[pathlib.Path]:
    """
    Process all expense files from the specified directory with integrated quality assessment.
    Quality assessment is performed on image files as a pre-filter before LlamaParse extraction.
    Both OpenCV and LLM quality assessments are run and saved to separate directories.

    Args:
        api_key: LlamaIndex API key
        input_folder: Directory containing expense files (PDFs, images)
        output_dir: Directory to save extracted markdown files

    Returns:
        List of paths to generated markdown files
    """
    logger.info(f"🚀 ***** Starting expense file processing with quality assessment...")
    
    api = LlamaIndexAPI(api_key)
    input_path = pathlib.Path(input_folder)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create quality assessment output directories
    quality_output_dir = pathlib.Path("quality_reports")
    quality_output_dir.mkdir(exist_ok=True)

    # Create separate LLM quality assessment output directory
    llm_quality_output_dir = pathlib.Path("llm_quality_reports")
    llm_quality_output_dir.mkdir(exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return []

    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning(f"No supported files found in {input_folder}")
        return []

    # Separate image files from non-image files
    image_files = []
    non_image_files = []
    
    for file_path in files:
        if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file_path)
        else:
            non_image_files.append(file_path)
    
    logger.info(f"📊 Found {len(files)} supported files: {len(image_files)} images, {len(non_image_files)} documents")
    
    # Initialize quality processor for image files
    quality_processor = None
    quality_results_summary = []
    llm_quality_results_summary = []

    if image_files:
        logger.info(f"🔍 ***** Step 1a: OpenCV Quality Assessment for {len(image_files)} image files...")
        quality_processor = ImageQualityProcessor(document_type='receipt')

        for i, file_path in enumerate(image_files, 1):
            logger.info(f"📸 Assessing image quality {i}/{len(image_files)}: {file_path.name}")
            quality_result = assess_image_quality_before_extraction(file_path, quality_processor, quality_output_dir)
            quality_results_summary.append({
                'filename': file_path.name,
                'quality_result': quality_result
            })

        # Separate LLM Quality Assessment with UQLM Validation
        logger.info(f"🤖 ***** Step 1b: LLM Quality Assessment with UQLM Validation for {len(image_files)} image files...")

        for i, file_path in enumerate(image_files, 1):
            logger.info(f"🤖 Assessing LLM image quality {i}/{len(image_files)}: {file_path.name}")

            # Find corresponding OpenCV assessment for validation comparison
            opencv_assessment = None
            for quality_summary in quality_results_summary:
                if quality_summary['filename'] == file_path.name:
                    opencv_assessment = quality_summary['quality_result']
                    break

            # Perform LLM assessment with UQLM validation
            llm_quality_result = assess_image_quality_llm_separate(
                file_path,
                llm_quality_output_dir,
                enable_validation=True,  # Enable UQLM validation
                opencv_assessment=opencv_assessment
            )
            llm_quality_results_summary.append({
                'filename': file_path.name,
                'llm_quality_result': llm_quality_result
            })
        
        # Log OpenCV quality assessment summary
        successful_quality_checks = [r for r in quality_results_summary if 'error' not in r['quality_result']]
        if successful_quality_checks:
            average_score = sum(r['quality_result']['quality_score'] for r in successful_quality_checks) / len(successful_quality_checks)
            passing_count = sum(1 for r in successful_quality_checks if r['quality_result']['quality_passed'])

            logger.info(f"✅ ***** OpenCV Quality Assessment Summary:")
            logger.info(f"    📊 Images assessed: {len(image_files)}")
            logger.info(f"    ✅ Successful assessments: {len(successful_quality_checks)}")
            logger.info(f"    🎯 Average quality score: {average_score:.1f}/100")
            logger.info(f"    📈 Images passing quality threshold: {passing_count}/{len(successful_quality_checks)}")

        # Log LLM quality assessment summary
        successful_llm_checks = [r for r in llm_quality_results_summary if 'error' not in r['llm_quality_result']]
        if successful_llm_checks:
                llm_average_score = sum(r['llm_quality_result']['quality_score'] for r in successful_llm_checks) / len(successful_llm_checks)
                llm_passing_count = sum(1 for r in successful_llm_checks if r['llm_quality_result']['quality_passed'])

        logger.info(f"🤖 ***** LLM Quality Assessment Summary:")
        logger.info(f"    � Images assessed: {len(image_files)}")
        logger.info(f"    ✅ Successful assessments: {len(successful_llm_checks)}")
        logger.info(f"    🎯 Average quality score: {llm_average_score:.1f}/100")
        logger.info(f"    📈 Images passing quality threshold: {llm_passing_count}/{len(successful_llm_checks)}")

        logger.info(f"�💾 OpenCV quality reports saved to: {quality_output_dir}/")
        logger.info(f"🤖 LLM quality reports saved to: {llm_quality_output_dir}/")
    
    # Step 2: LlamaParse extraction for all files (images + documents)
    logger.info(f"🔄 ***** Step 2: LlamaParse extraction for all {len(files)} files...")
    generated_files = []

    for i, file_path in enumerate(files, 1):
        file_type = "image" if file_path in image_files else "document"
        logger.info(f"🔄 Processing {file_type} {i}/{len(files)}: {file_path.name}")
        
        output_file = output_path / (file_path.stem + '.md')
        extract_markdown(file_path, api, output_path)
        if output_file.exists():
            generated_files.append(output_file)
            logger.info(f"✅ Successfully processed {file_path.name}")
        else:
            logger.warning(f"⚠️ Failed to process {file_path.name}")

    # Final summary
    logger.info(f"✅ ***** Processing completed:")
    logger.info(f"    📄 LlamaParse extraction: {len(generated_files)}/{len(files)} files processed successfully")
    if image_files:
        logger.info(f"    🔍 Quality assessments: {len(quality_results_summary)} image files assessed")
        logger.info(f"    💾 Quality reports available in: {quality_output_dir}/")
    
    return generated_files