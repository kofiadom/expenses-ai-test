import requests
import pathlib
import time
from agno.utils.log import logger

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

def extract_markdown(file_path: pathlib.Path, api: LlamaIndexAPI, output_dir: pathlib.Path):
    logger.info(f"Uploading {file_path} to LlamaParse...")
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
    logger.info(f"Successfully saved markdown to {output_file}")

def process_expense_files(api_key: str, input_folder: str = "expense_files", output_dir: str = "llamaparse_output") -> list[pathlib.Path]:
    """
    Process all expense files from the specified directory and return list of generated markdown files.

    Args:
        api_key: LlamaIndex API key
        input_folder: Directory containing expense files (PDFs, images)
        output_dir: Directory to save extracted markdown files

    Returns:
        List of paths to generated markdown files
    """
    api = LlamaIndexAPI(api_key)
    input_path = pathlib.Path(input_folder)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return []

    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning(f"No supported files found in {input_folder}")
        return []

    logger.info(f"Found {len(files)} supported files in {input_folder}. Starting extraction...")
    generated_files = []

    for file_path in files:
        output_file = output_path / (file_path.stem + '.md')
        extract_markdown(file_path, api, output_path)
        if output_file.exists():
            generated_files.append(output_file)
            logger.info(f"Successfully processed {file_path.name}")
        else:
            logger.warning(f"Failed to process {file_path.name}")

    logger.info(f"Extraction completed: {len(generated_files)}/{len(files)} files processed successfully")
    return generated_files