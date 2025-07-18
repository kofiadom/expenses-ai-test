#!/usr/bin/env python3
"""
Single File Extraction and Compliance Test Script

This script allows you to test extraction and compliance analysis on a single file
with configurable parameters for filename, country, and ICP.

Usage:
    python test_single_file_extraction.py
    
Then modify the placeholders in the script:
    - FILENAME: Name of the file to test (without extension)
    - COUNTRY: Country for compliance rules (e.g., "Germany", "Switzerland")
    - ICP: ICP name (e.g., "Global People", "goGlobal", "Parakar", "Atlas")
"""

import asyncio
import json
import os
import pathlib
import time
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from agno.utils.log import logger

# Import the required agents and functions
from llamaparse_extractor import process_expense_files
from file_classification_agent import classify_file
from data_extraction_agent import extract_data_from_receipt
from issue_detection_agent import analyze_compliance_issues

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION PLACEHOLDERS - MODIFY THESE VALUES
# ============================================================================

FILENAME = "switzerland"  # File name without extension (e.g., "receipt_001", "german_file_2")
COUNTRY = "Switzerland"   # Country for compliance rules (e.g., "Germany", "Switzerland")
ICP = "Global People"     # ICP name: "Global People", "goGlobal", "Parakar", "Atlas"

# Optional: Override default directories
INPUT_FOLDER = "expense_files"      # Directory containing the original file
RESULTS_DIR = "single_file_results" # Directory to save results

# ============================================================================

class SingleFileExtractor:
    """Test extraction and compliance on a single file."""
    
    def __init__(self, filename: str, country: str, icp: str, 
                 input_folder: str = "expense_files", 
                 results_dir: str = "single_file_results"):
        """Initialize the single file extractor."""
        self.filename = filename
        self.country = country
        self.icp = icp
        self.input_folder = pathlib.Path(input_folder)
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Get API key
        self.llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
        if not self.llamaparse_api_key:
            raise ValueError("LLAMAPARSE_API_KEY not found in environment variables")
        
        logger.info(f"🔧 Initialized Single File Extractor")
        logger.info(f"   File: {self.filename}")
        logger.info(f"   Country: {self.country}")
        logger.info(f"   ICP: {self.icp}")
        logger.info(f"   Input folder: {self.input_folder}")
        logger.info(f"   Results directory: {self.results_dir}")
    
    async def process_single_file(self) -> Dict[str, Any]:
        """Process a single file through the complete workflow."""
        
        logger.info(f"\n🚀 Starting Single File Processing")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Find the source file
            source_file = self._find_source_file()
            if not source_file:
                raise FileNotFoundError(f"Source file not found for: {self.filename}")
            
            logger.info(f"📁 Found source file: {source_file}")
            
            # Step 2: Extract to markdown using LlamaParse
            logger.info(f"🔄 Step 1: Extracting document using LlamaParse...")
            markdown_file = await self._extract_to_markdown(source_file)
            
            # Step 3: Load markdown content
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            logger.info(f"📄 Loaded markdown content: {len(markdown_content)} characters")
            
            # Step 4: Run classification and extraction concurrently
            logger.info(f"🔄 Step 2: Running classification and extraction...")
            classification_result, extraction_result = await self._run_agents(markdown_content)
            
            # Step 5: Run compliance analysis
            logger.info(f"🔄 Step 3: Running compliance analysis...")
            compliance_result = await self._run_compliance_analysis(extraction_result)
            
            # Step 6: Compile results
            total_time = time.time() - start_time
            
            result = {
                "file_name": self.filename,
                "source_file": str(source_file),
                "markdown_file": str(markdown_file),
                "country": self.country,
                "icp": self.icp,
                "classification_result": classification_result,
                "extraction_result": extraction_result,
                "compliance_result": compliance_result,
                "processing_time": {
                    "total_seconds": round(total_time, 2)
                },
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            # Step 7: Save results
            self._save_results(result)
            
            logger.info(f"✅ Processing completed in {total_time:.2f} seconds")
            logger.info(f"📊 Results saved to: {self.results_dir}")
            
            return result
            
        except Exception as e:
            error_result = {
                "file_name": self.filename,
                "country": self.country,
                "icp": self.icp,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.error(f"❌ Processing failed: {str(e)}")
            self._save_results(error_result)
            
            return error_result

    def _find_source_file(self) -> Optional[pathlib.Path]:
        """Find the source file with various extensions."""
        # Common file extensions
        extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif', '.doc', '.docx']

        # Search in input folder
        for ext in extensions:
            file_path = self.input_folder / f"{self.filename}{ext}"
            if file_path.exists():
                return file_path

        # Also search in dataset directory
        dataset_dir = pathlib.Path("dataset")
        if dataset_dir.exists():
            for ext in extensions:
                file_path = dataset_dir / f"{self.filename}{ext}"
                if file_path.exists():
                    return file_path

        return None

    async def _extract_to_markdown(self, source_file: pathlib.Path) -> pathlib.Path:
        """Extract the source file to markdown using LlamaParse."""
        # Create a temporary directory for this single file
        temp_dir = self.results_dir / "temp_extraction"
        temp_dir.mkdir(exist_ok=True)

        # Copy the source file to temp directory
        temp_file = temp_dir / source_file.name
        import shutil
        shutil.copy2(source_file, temp_file)

        # Run LlamaParse extraction
        markdown_files = await process_expense_files(
            api_key=self.llamaparse_api_key,
            input_folder=str(temp_dir),
            output_dir=str(self.results_dir / "markdown_output")
        )

        if not markdown_files:
            raise RuntimeError("LlamaParse extraction failed - no markdown files generated")

        # Find the markdown file for our specific file
        expected_markdown = None
        for md_file in markdown_files:
            if md_file.stem == source_file.stem:
                expected_markdown = md_file
                break

        if not expected_markdown:
            raise RuntimeError(f"Markdown file not found for {source_file.name}")

        return expected_markdown

    async def _run_agents(self, markdown_content: str) -> tuple:
        """Run classification and extraction agents concurrently."""
        # Load compliance data for extraction
        compliance_file = pathlib.Path(f"data/{self.country.lower()}.json")
        if compliance_file.exists():
            with open(compliance_file, 'r') as f:
                compliance_data = json.load(f)
            logger.info(f"📋 Loaded compliance data for {self.country}")
        else:
            compliance_data = {}
            logger.warning(f"⚠️ No compliance data found for {self.country}")

        # Run classification and extraction concurrently
        classification_task = asyncio.create_task(
            self._run_classification(markdown_content)
        )
        extraction_task = asyncio.create_task(
            self._run_extraction(compliance_data, markdown_content)
        )

        classification_result, extraction_result = await asyncio.gather(
            classification_task, extraction_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(classification_result, Exception):
            logger.error(f"Classification failed: {str(classification_result)}")
            classification_result = {"error": str(classification_result)}

        if isinstance(extraction_result, Exception):
            logger.error(f"Extraction failed: {str(extraction_result)}")
            extraction_result = {"error": str(extraction_result)}

        return classification_result, extraction_result

    async def _run_classification(self, markdown_content: str) -> Dict:
        """Run file classification."""
        try:
            start_time = time.time()
            logger.info(f"🔍 Running classification agent...")

            result = classify_file(markdown_content, self.country)

            # Parse the result if it's a string
            if isinstance(result, str):
                result = json.loads(result)

            classification_time = time.time() - start_time
            logger.info(f"✅ Classification completed in {classification_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {"error": str(e)}

    async def _run_extraction(self, compliance_data: Dict, markdown_content: str) -> Dict:
        """Run data extraction."""
        try:
            start_time = time.time()
            logger.info(f"📊 Running extraction agent...")

            result = extract_data_from_receipt(
                json.dumps(compliance_data),
                markdown_content
            )

            # Parse the result if it's a string
            if isinstance(result, str):
                result = json.loads(result)

            extraction_time = time.time() - start_time
            logger.info(f"✅ Extraction completed in {extraction_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return {"error": str(e)}

    async def _run_compliance_analysis(self, extraction_result: Dict) -> Dict:
        """Run compliance analysis."""
        try:
            # Skip compliance if extraction failed
            if "error" in extraction_result:
                logger.warning("⚠️ Skipping compliance analysis due to extraction error")
                return {"error": "Skipped due to extraction failure"}

            start_time = time.time()
            logger.info(f"⚖️ Running compliance analysis...")

            # Load compliance data
            compliance_file = pathlib.Path(f"data/{self.country.lower()}.json")
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    compliance_data = json.load(f)
            else:
                compliance_data = {}
                logger.warning(f"No compliance data found for {self.country}")

            # Get receipt type from extraction result (default to "All")
            receipt_type = "All"
            if "expense_type" in extraction_result:
                receipt_type = extraction_result["expense_type"] or "All"

            # Run compliance analysis
            result = await analyze_compliance_issues(
                self.country, receipt_type, self.icp, compliance_data, extraction_result
            )

            # Parse the result if it's a string
            if isinstance(result, str):
                result = json.loads(result)

            compliance_time = time.time() - start_time
            logger.info(f"✅ Compliance analysis completed in {compliance_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Compliance analysis error: {str(e)}")
            return {"error": str(e)}

    def _save_results(self, result: Dict):
        """Save results to JSON file."""
        try:
            # Save main result file
            result_file = self.results_dir / f"{self.filename}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Main result saved: {result_file}")

            # Save individual component files for easier analysis
            if "classification_result" in result and result["classification_result"]:
                classification_file = self.results_dir / f"{self.filename}_classification.json"
                with open(classification_file, 'w', encoding='utf-8') as f:
                    json.dump(result["classification_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Classification saved: {classification_file}")

            if "extraction_result" in result and result["extraction_result"]:
                extraction_file = self.results_dir / f"{self.filename}_extraction.json"
                with open(extraction_file, 'w', encoding='utf-8') as f:
                    json.dump(result["extraction_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Extraction saved: {extraction_file}")

            if "compliance_result" in result and result["compliance_result"]:
                compliance_file = self.results_dir / f"{self.filename}_compliance.json"
                with open(compliance_file, 'w', encoding='utf-8') as f:
                    json.dump(result["compliance_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Compliance saved: {compliance_file}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def print_summary(self, result: Dict):
        """Print a summary of the processing results."""
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"File: {result.get('file_name', 'Unknown')}")
        print(f"Country: {result.get('country', 'Unknown')}")
        print(f"ICP: {result.get('icp', 'Unknown')}")
        print(f"Status: {result.get('status', 'Unknown')}")

        if result.get('status') == 'completed':
            print(f"Processing Time: {result.get('processing_time', {}).get('total_seconds', 0):.2f} seconds")

            # Classification summary
            classification = result.get('classification_result', {})
            if classification and 'error' not in classification:
                print(f"\n🔍 CLASSIFICATION:")
                print(f"  Is Expense: {classification.get('is_expense', 'Unknown')}")
                print(f"  Expense Type: {classification.get('expense_type', 'Unknown')}")
                print(f"  Language: {classification.get('language', 'Unknown')}")
                print(f"  Confidence: {classification.get('language_confidence', 0):.2f}")

            # Extraction summary
            extraction = result.get('extraction_result', {})
            if extraction and 'error' not in extraction:
                print(f"\n📊 EXTRACTION:")
                print(f"  Vendor: {extraction.get('vendor_name', 'Unknown')}")
                print(f"  Total Amount: {extraction.get('total_amount', 'Unknown')}")
                print(f"  Date: {extraction.get('transaction_date', 'Unknown')}")
                print(f"  Currency: {extraction.get('currency', 'Unknown')}")

            # Compliance summary
            compliance = result.get('compliance_result', {})
            if compliance and 'error' not in compliance:
                issues = compliance.get('issues', [])
                print(f"\n⚖️ COMPLIANCE:")
                print(f"  Issues Found: {len(issues)}")
                if issues:
                    for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues
                        print(f"    {i}. {issue.get('issue_type', 'Unknown')}: {issue.get('description', 'No description')[:80]}...")
                    if len(issues) > 3:
                        print(f"    ... and {len(issues) - 3} more issues")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        print(f"\n📁 Results saved to: {self.results_dir}")
        print("=" * 60)


async def main():
    """Main function to run the single file extraction test."""

    print(f"🚀 Single File Extraction and Compliance Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  File: {FILENAME}")
    print(f"  Country: {COUNTRY}")
    print(f"  ICP: {ICP}")
    print(f"  Input Folder: {INPUT_FOLDER}")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 60)

    try:
        # Create extractor
        extractor = SingleFileExtractor(
            filename=FILENAME,
            country=COUNTRY,
            icp=ICP,
            input_folder=INPUT_FOLDER,
            results_dir=RESULTS_DIR
        )

        # Process the file
        result = await extractor.process_single_file()

        # Print summary
        extractor.print_summary(result)

        return 0

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

    async def _run_compliance_analysis(self, extraction_result: Dict) -> Dict:
        """Run compliance analysis."""
        try:
            # Skip compliance if extraction failed
            if "error" in extraction_result:
                logger.warning("⚠️ Skipping compliance analysis due to extraction error")
                return {"error": "Skipped due to extraction failure"}

            start_time = time.time()
            logger.info(f"⚖️ Running compliance analysis...")

            # Load compliance data
            compliance_file = pathlib.Path(f"data/{self.country.lower()}.json")
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    compliance_data = json.load(f)
            else:
                compliance_data = {}
                logger.warning(f"No compliance data found for {self.country}")

            # Get receipt type from extraction result (default to "All")
            receipt_type = "All"
            if "expense_type" in extraction_result:
                receipt_type = extraction_result["expense_type"] or "All"

            # Run compliance analysis
            result = await analyze_compliance_issues(
                self.country, receipt_type, self.icp,
                compliance_data, extraction_result
            )

            # Parse the result if it's a string
            if isinstance(result, str):
                result = json.loads(result)

            compliance_time = time.time() - start_time
            logger.info(f"✅ Compliance analysis completed in {compliance_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Compliance analysis error: {str(e)}")
            return {"error": str(e)}

    def _save_results(self, result: Dict):
        """Save results to JSON file."""
        try:
            # Save main result
            result_file = self.results_dir / f"{self.filename}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Main result saved: {result_file}")

            # Save individual components for easier analysis
            if "classification_result" in result and result["classification_result"]:
                classification_file = self.results_dir / f"{self.filename}_classification.json"
                with open(classification_file, 'w', encoding='utf-8') as f:
                    json.dump(result["classification_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Classification saved: {classification_file}")

            if "extraction_result" in result and result["extraction_result"]:
                extraction_file = self.results_dir / f"{self.filename}_extraction.json"
                with open(extraction_file, 'w', encoding='utf-8') as f:
                    json.dump(result["extraction_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Extraction saved: {extraction_file}")

            if "compliance_result" in result and result["compliance_result"]:
                compliance_file = self.results_dir / f"{self.filename}_compliance.json"
                with open(compliance_file, 'w', encoding='utf-8') as f:
                    json.dump(result["compliance_result"], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Compliance saved: {compliance_file}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def print_summary(self, result: Dict):
        """Print a summary of the processing results."""
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"File: {result.get('file_name', 'Unknown')}")
        print(f"Country: {result.get('country', 'Unknown')}")
        print(f"ICP: {result.get('icp', 'Unknown')}")
        print(f"Status: {result.get('status', 'Unknown')}")

        if result.get('status') == 'completed':
            print(f"Processing Time: {result.get('processing_time', {}).get('total_seconds', 0):.2f} seconds")

            # Classification summary
            classification = result.get('classification_result', {})
            if classification and 'error' not in classification:
                print(f"\n🔍 CLASSIFICATION:")
                print(f"  Is Expense: {classification.get('is_expense', 'Unknown')}")
                print(f"  Expense Type: {classification.get('expense_type', 'Unknown')}")
                print(f"  Language: {classification.get('language', 'Unknown')}")
                print(f"  Confidence: {classification.get('language_confidence', 0):.2f}")

            # Extraction summary
            extraction = result.get('extraction_result', {})
            if extraction and 'error' not in extraction:
                print(f"\n📊 EXTRACTION:")
                print(f"  Vendor: {extraction.get('vendor_name', 'Unknown')}")
                print(f"  Total Amount: {extraction.get('total_amount', 'Unknown')}")
                print(f"  Date: {extraction.get('transaction_date', 'Unknown')}")
                print(f"  Currency: {extraction.get('currency', 'Unknown')}")

            # Compliance summary
            compliance = result.get('compliance_result', {})
            if compliance and 'error' not in compliance:
                issues = compliance.get('issues', [])
                print(f"\n⚖️ COMPLIANCE:")
                print(f"  Issues Found: {len(issues)}")
                if issues:
                    for issue in issues[:3]:  # Show first 3 issues
                        print(f"    - {issue.get('issue_type', 'Unknown')}: {issue.get('description', 'No description')}")
                    if len(issues) > 3:
                        print(f"    ... and {len(issues) - 3} more issues")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        print(f"\n📁 Results saved to: {self.results_dir}")
        print("=" * 60)


async def main():
    """Main function to run the single file extraction test."""

    print(f"🚀 Single File Extraction and Compliance Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  File: {FILENAME}")
    print(f"  Country: {COUNTRY}")
    print(f"  ICP: {ICP}")
    print(f"  Input Folder: {INPUT_FOLDER}")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 60)

    try:
        # Create extractor
        extractor = SingleFileExtractor(
            filename=FILENAME,
            country=COUNTRY,
            icp=ICP,
            input_folder=INPUT_FOLDER,
            results_dir=RESULTS_DIR
        )

        # Process the file
        result = await extractor.process_single_file()

        # Print summary
        extractor.print_summary(result)

        return 0

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
