"""
Professional Expense Processing Workflow
"""

import asyncio
import json
import pathlib
from datetime import datetime
from typing import Dict, List, AsyncGenerator

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.team import Team
from agno.workflow import Workflow, RunResponse
from agno.utils.log import logger
from dotenv import load_dotenv

from llamaparse_extractor import process_expense_files
from file_classification_agent import classify_file
from data_extraction_agent import extract_data_from_receipt
from issue_detection_agent import analyze_compliance_issues

# Load environment variables
load_dotenv()

class ExpenseProcessingWorkflow(Workflow):
    """Professional expense processing workflow."""

    description: str = "Expense document processing workflow with multi-agent coordination"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.processing_team = Team(
            name="Expense Processing Team",
            mode="coordinate",
            model=OpenAIChat(id="gpt-4o"),
            #model=Claude(id="claude-3-7-sonnet-20250219"),
            members=[
                Agent(
                    name="Classification Agent",
                    role="Classify expense documents",
                    model=OpenAIChat(id="gpt-4o"),
                    #model=Claude(id="claude-3-7-sonnet-20250219"),
                    instructions="Analyze documents to determine if they are valid expenses."
                ),
                Agent(
                    name="Data Extraction Agent",
                    role="Extract structured data from expense documents",
                    model=OpenAIChat(id="gpt-4o"),
                    #model=Claude(id="claude-3-7-sonnet-20250219"),
                    instructions="Extract structured data from expense documents."
                )
            ],
            instructions=["Process expense documents concurrently"],
            show_tool_calls=True,
            markdown=True
        )

    async def process_expenses(
        self,
        country: str,
        icp: str,
        llamaparse_api_key: str,
        input_folder: str = "expense_files"
    ) -> AsyncGenerator[RunResponse, None]:
        """
        Execute the complete expense processing workflow.

        Args:
            country: Country for compliance rules (e.g., "Germany")
            icp: ICP name (e.g., "Global People", "goGlobal", "Parakar", "Atlas")
            llamaparse_api_key: API key for LlamaParse document extraction
            input_folder: Directory containing expense files

        Yields:
            RunResponse objects with processing updates and final results
        """
        logger.info(f"Starting expense processing workflow for {country}/{icp}")
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Debug mode: {self.debug_mode}")

        try:
            # Step 1: Extract documents to markdown
            yield RunResponse(
                content="🔄 Step 1: Extracting documents using LlamaParse..."
            )

            logger.info(f"Extracting documents from {input_folder} using LlamaParse")
            markdown_files = await self._extract_documents(llamaparse_api_key, input_folder)
            if not markdown_files:
                logger.warning("No documents found or extraction failed")
                yield RunResponse(
                    content="❌ No documents found or extraction failed"
                )
                # Save empty results to session state
                self.session_state["processing_results"] = []
                self.session_state["summary"] = "No documents found to process"
                return

            logger.info(f"Successfully extracted {len(markdown_files)} documents")
            yield RunResponse(
                content=f"✅ Successfully extracted {len(markdown_files)} documents"
            )

            # Step 2: Process each document through the pipeline
            logger.info(f"Starting processing of {len(markdown_files)} documents")
            results = []
            for i, markdown_file in enumerate(markdown_files, 1):
                logger.info(f"Processing document {i}/{len(markdown_files)}: {markdown_file.name}")
                yield RunResponse(
                    content=f"🔄 Step 2: Processing document {i}/{len(markdown_files)}: {markdown_file.name}"
                )

                result = await self._process_single_document(
                    markdown_file, country, icp
                )
                results.append(result)

                logger.info(f"Completed processing {markdown_file.name} - Status: {result.get('status', 'unknown')}")
                yield RunResponse(
                    content=f"✅ Completed processing {markdown_file.name}"
                )

            # Step 3: Save individual results and generate summary
            logger.info("Saving individual results and generating summary")
            yield RunResponse(
                content="🔄 Step 3: Saving results and generating summary..."
            )

            self._save_individual_results(results)
            summary = self._generate_summary(results)
            logger.info(f"Processing summary: {summary}")

            # Save results to session state
            self.session_state["processing_results"] = results
            self.session_state["summary"] = summary

            yield RunResponse(
                content=summary
            )

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            yield RunResponse(
                content=f"❌ Workflow failed: {str(e)}"
            )
            raise

    async def _extract_documents(self, api_key: str, input_folder: str) -> List[pathlib.Path]:
        """Extract documents using LlamaParse."""
        try:
            # Run document extraction
            markdown_files = process_expense_files(api_key, input_folder)
            return markdown_files
        except Exception as e:
            logger.error(f"Document extraction failed: {str(e)}")
            return []

    async def _process_single_document(
        self,
        markdown_file: pathlib.Path,
        country: str,
        icp: str
    ) -> Dict:
        """Process a single document through classification, extraction, and analysis."""
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            logger.info(f"Processing {markdown_file.name} - Markdown length: {len(markdown_content)} characters")

            # Step 2a & 2b: Run classification and extraction concurrently
            classification_task = self._classify_document(markdown_content, country)
            extraction_task = self._extract_data(markdown_content, country)

            classification_result, extraction_result = await asyncio.gather(
                classification_task, extraction_task, return_exceptions=True
            )

            if isinstance(classification_result, Exception):
                logger.error(f"Classification failed: {str(classification_result)}")
                classification_result = {"error": str(classification_result)}
            else:
                logger.info(f"Classification completed for {markdown_file.name}")

            if isinstance(extraction_result, Exception):
                logger.error(f"Extraction failed: {str(extraction_result)}")
                extraction_result = {"error": str(extraction_result)}
            else:
                logger.info(f"Data extraction completed for {markdown_file.name}")

            # Step 2c: Compliance analysis (only if extraction succeeded)
            compliance_result = {}
            if not isinstance(extraction_result, dict) or "error" not in extraction_result:
                try:
                    # Get the expense type from classification result
                    expense_type = "All"  # Default fallback
                    if isinstance(classification_result, dict) and "expense_type" in classification_result:
                        expense_type = classification_result["expense_type"] or "All"

                    logger.info(f"Starting compliance analysis for {markdown_file.name} (type: {expense_type})")
                    compliance_result = await self._analyze_compliance(
                        extraction_result, country, icp, expense_type
                    )
                    logger.info(f"Compliance analysis completed for {markdown_file.name}")
                except Exception as e:
                    logger.error(f"Compliance analysis failed: {str(e)}")
                    compliance_result = {"error": str(e)}
            else:
                logger.warning(f"Skipping compliance analysis for {markdown_file.name} due to extraction failure")

            return {
                "file_name": markdown_file.name,
                "classification": classification_result,
                "extraction": extraction_result,
                "compliance": compliance_result,
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Document processing failed for {markdown_file.name}: {str(e)}")
            return {
                "file_name": markdown_file.name,
                "error": str(e),
                "status": "failed"
            }

    async def _classify_document(self, markdown_content: str, country: str) -> Dict:
        """Classify document using file classification agent."""
        try:
            logger.debug(f"Sending markdown to classification agent - Length: {len(markdown_content)}")
            result = classify_file(markdown_content, country)

            # Handle different response formats
            if hasattr(result, 'content'):
                content = result.content
                if content is None or content.strip() == "":
                    logger.error("Empty content returned from classification agent")
                    return {"error": "Empty response from classification agent"}

                # Handle markdown-wrapped JSON
                content = content.strip()
                if content.startswith('```json') and content.endswith('```'):
                    content = content[7:-3].strip()  # Remove ```json and ```
                elif content.startswith('```') and content.endswith('```'):
                    content = content[3:-3].strip()  # Remove ``` and ```

                parsed_result = json.loads(content)
            else:
                # If result doesn't have content attribute, assume it's already parsed
                parsed_result = result

            logger.debug(f"Classification result: {parsed_result}")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in classification: {str(e)}")
            logger.error(f"Raw content: {getattr(result, 'content', 'No content attribute') if 'result' in locals() else 'No result'}")
            return {"error": f"Invalid JSON response from classification agent: {str(e)}"}
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {"error": str(e)}

    async def _extract_data(self, markdown_content: str, country: str) -> Dict:
        """Extract data using data extraction agent."""
        try:
            compliance_file = pathlib.Path(f"data/{country.lower()}.json")
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    compliance_data = json.load(f)
                compliance_json = json.dumps(compliance_data)
                logger.debug(f"Loaded compliance data for {country} - {len(compliance_data)} sections")
            else:
                compliance_json = "{}"
                logger.warning(f"No compliance data found for {country}")

            logger.debug(f"Sending markdown to data extraction agent - Length: {len(markdown_content)}")
            result = extract_data_from_receipt(compliance_json, markdown_content)

            # Handle different response formats
            if hasattr(result, 'content'):
                content = result.content
                if content is None or content.strip() == "":
                    logger.error("Empty content returned from data extraction agent")
                    return {"error": "Empty response from data extraction agent"}

                # Handle markdown-wrapped JSON
                content = content.strip()
                if content.startswith('```json') and content.endswith('```'):
                    content = content[7:-3].strip()  # Remove ```json and ```
                elif content.startswith('```') and content.endswith('```'):
                    content = content[3:-3].strip()  # Remove ``` and ```

                parsed_result = json.loads(content)
            else:
                # If result doesn't have content attribute, assume it's already parsed
                parsed_result = result

            logger.debug(f"Extraction result keys: {list(parsed_result.keys()) if isinstance(parsed_result, dict) else 'Not a dict'}")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in data extraction: {str(e)}")
            logger.error(f"Raw content: {getattr(result, 'content', 'No content attribute') if 'result' in locals() else 'No result'}")
            return {"error": f"Invalid JSON response from data extraction agent: {str(e)}"}
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            return {"error": str(e)}

    async def _analyze_compliance(
        self,
        extraction_result: Dict,
        country: str,
        icp: str,
        receipt_type: str
    ) -> Dict:
        """Analyze compliance using issue detection agent."""
        try:
            compliance_file = pathlib.Path(f"data/{country.lower()}.json")
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    compliance_data = json.load(f)
                logger.debug(f"Loaded compliance data for analysis - {len(compliance_data)} sections")
            else:
                compliance_data = {}
                logger.warning(f"No compliance data found for {country}")

            logger.debug(f"Sending extracted data to compliance agent - Keys: {list(extraction_result.keys())}")
            result = analyze_compliance_issues(
                country, receipt_type, icp, compliance_data, extraction_result
            )

            # Handle different response formats
            if hasattr(result, 'content'):
                content = result.content
                if content is None or content.strip() == "":
                    logger.error("Empty content returned from compliance analysis agent")
                    return {"error": "Empty response from compliance analysis agent"}

                # Handle markdown-wrapped JSON
                content = content.strip()
                if content.startswith('```json') and content.endswith('```'):
                    content = content[7:-3].strip()  # Remove ```json and ```
                elif content.startswith('```') and content.endswith('```'):
                    content = content[3:-3].strip()  # Remove ``` and ```

                parsed_result = json.loads(content)
                logger.debug(f"Compliance analysis result: {parsed_result}")
                return parsed_result
            else:
                # If result doesn't have content attribute, assume it's already parsed
                parsed_result = result
                logger.debug(f"Compliance analysis result (no content attr): {parsed_result}")
                return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in compliance analysis: {str(e)}")
            logger.error(f"Raw content: {getattr(result, 'content', 'No content attribute') if 'result' in locals() else 'No result'}")
            return {"error": f"Invalid JSON response from compliance analysis agent: {str(e)}"}
        except Exception as e:
            logger.error(f"Compliance analysis error: {str(e)}")
            return {"error": str(e)}

    def _save_individual_results(self, results: List[Dict]) -> None:
        """Save individual processing results to separate JSON files."""
        results_dir = pathlib.Path("results")
        results_dir.mkdir(exist_ok=True)

        for result in results:
            if result.get("status") == "completed":
                file_name = result.get("file_name", "unknown")
                # Remove .md extension and add .json
                base_name = pathlib.Path(file_name).stem
                output_file = results_dir / f"{base_name}.json"

                # Create comprehensive result structure
                individual_result = {
                    "source_file": file_name,
                    "processing_timestamp": datetime.now().isoformat(),
                    "classification_result": result.get("classification", {}),
                    "extraction_result": result.get("extraction", {}),
                    "compliance_result": result.get("compliance", {}),
                    "processing_status": result.get("status", "unknown")
                }

                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(individual_result, f, indent=2)
                    logger.info(f"Saved individual result: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save result for {file_name}: {e}")
            else:
                logger.warning(f"Skipping result save for failed processing: {result.get('file_name', 'unknown')}")

    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate processing summary."""
        total_files = len(results)
        successful = len([r for r in results if r.get("status") == "completed"])
        failed = total_files - successful

        expenses = 0
        for result in results:
            classification = result.get("classification", {})
            if isinstance(classification, dict) and classification.get("is_expense"):
                expenses += 1

        return f"Processed {total_files} files: {successful} successful, {failed} failed, {expenses} valid expenses"
