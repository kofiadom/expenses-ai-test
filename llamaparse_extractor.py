import requests
import pathlib
import time
import json
from typing import Dict, List, Tuple
from agno.utils.log import logger

from image_quality_processor import ImageQualityProcessor, SUPPORTED_IMAGE_EXTENSIONS

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
    Assess image quality before LlamaParse extraction and save results
    
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
    
    # Create quality assessment output directory
    quality_output_dir = pathlib.Path("quality_reports")
    quality_output_dir.mkdir(exist_ok=True)

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
    
    if image_files:
        logger.info(f"🔍 ***** Step 1a: Quality Assessment for {len(image_files)} image files...")
        quality_processor = ImageQualityProcessor(document_type='receipt')
        
        for i, file_path in enumerate(image_files, 1):
            logger.info(f"📸 Assessing image quality {i}/{len(image_files)}: {file_path.name}")
            quality_result = assess_image_quality_before_extraction(file_path, quality_processor, quality_output_dir)
            quality_results_summary.append({
                'filename': file_path.name,
                'quality_result': quality_result
            })
        
        # Log quality assessment summary
        successful_quality_checks = [r for r in quality_results_summary if 'error' not in r['quality_result']]
        if successful_quality_checks:
            average_score = sum(r['quality_result']['quality_score'] for r in successful_quality_checks) / len(successful_quality_checks)
            passing_count = sum(1 for r in successful_quality_checks if r['quality_result']['quality_passed'])
            
            logger.info(f"✅ ***** Quality assessment summary:")
            logger.info(f"    📊 Images assessed: {len(image_files)}")
            logger.info(f"    ✅ Successful assessments: {len(successful_quality_checks)}")
            logger.info(f"    🎯 Average quality score: {average_score:.1f}/100")
            logger.info(f"    📈 Images passing quality threshold: {passing_count}/{len(successful_quality_checks)}")
        
        logger.info(f"💾 Quality reports saved to: {quality_output_dir}/")
    
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