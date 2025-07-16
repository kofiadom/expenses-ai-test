"""
Professional Expense Processing Workflow
"""

import asyncio
import json
import pathlib
import time
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
from dataset_utils import load_dataset_entries, validate_dataset_entry
from citation_generator import generate_citations, get_citation_stats

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
        dataset_dir: str = "dataset",
        llamaparse_api_key: str = None,
        input_folder: str = "expense_files"
    ) -> AsyncGenerator[RunResponse, None]:
        """
        Execute the complete expense processing workflow using dataset metadata.

        Args:
            dataset_dir: Directory containing dataset JSON files with metadata
            llamaparse_api_key: API key for LlamaParse document extraction
            input_folder: Directory containing expense files

        Yields:
            RunResponse objects with processing updates and final results
        """
        workflow_start_time = time.time()
        logger.info(f"🕐 Starting expense processing workflow using dataset: {dataset_dir}")
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

            # Step 2: Load dataset entries and process each with its metadata
            dataset_entries = load_dataset_entries(dataset_dir)
            if not dataset_entries:
                logger.warning("No dataset entries found")
                yield RunResponse(
                    content="❌ No dataset entries found"
                )
                self.session_state["processing_results"] = []
                self.session_state["summary"] = "No dataset entries found to process"
                return

            logger.info(f"Starting processing of {len(dataset_entries)} dataset entries")
            results = []
            
            for i, entry in enumerate(dataset_entries, 1):
                # Convert filepath to markdown filename: "expense_files/austrian_file.png" -> "austrian_file.md"
                base_name = pathlib.Path(entry['filepath']).stem
                markdown_file = pathlib.Path(f"llamaparse_output/{base_name}.md")
                
                logger.info(f"Processing entry {i}/{len(dataset_entries)}: {entry['filepath']} -> {markdown_file.name}")
                yield RunResponse(
                    content=f"🔄 Step 2: Processing entry {i}/{len(dataset_entries)}: {entry['filepath']} (Country: {entry['country']}, ICP: {entry['icp']})"
                )

                if markdown_file.exists():
                    result = await self._process_single_document(
                        markdown_file, entry['country'], entry['icp']
                    )
                    result['dataset_metadata'] = entry
                    results.append(result)

                    # Save individual result immediately after processing
                    self._save_single_result(result)

                    logger.info(f"Completed processing {markdown_file.name} - Status: {result.get('status', 'unknown')}")
                    yield RunResponse(
                        content=f"✅ Completed processing {markdown_file.name}"
                    )
                else:
                    logger.warning(f"Markdown file not found for {entry['filepath']}: {markdown_file}")
                    yield RunResponse(
                        content=f"⚠️ Markdown file not found for {entry['filepath']}"
                    )
                    # Add failed result for missing markdown file
                    results.append({
                        "file_name": f"{base_name}.md",
                        "error": f"Markdown file not found: {markdown_file}",
                        "status": "failed",
                        "dataset_metadata": entry
                    })

            # Step 3: Generate summary (individual results already saved incrementally)
            logger.info("Generating processing summary")
            yield RunResponse(
                content="🔄 Step 3: Generating processing summary..."
            )

            save_start_time = time.time()
            self._save_individual_results(results)  # Legacy call - does minimal work now
            summary = self._generate_summary(results)
            save_time = time.time() - save_start_time
            logger.info(f"⏱️ Summary generation completed in {save_time:.2f} seconds")
            logger.info(f"Processing summary: {summary}")

            # Save results to session state
            self.session_state["processing_results"] = results
            self.session_state["summary"] = summary

            total_workflow_time = time.time() - workflow_start_time
            logger.info(f"⏱️ Total workflow completed in {total_workflow_time:.2f} seconds")

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
            start_time = time.time()
            logger.info("🕐 Starting document extraction timing")

            # Run document extraction
            markdown_files = process_expense_files(api_key, input_folder)

            extraction_time = time.time() - start_time
            logger.info(f"⏱️ Document extraction completed in {extraction_time:.2f} seconds")

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
            document_start_time = time.time()

            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            logger.info(f"🕐 Processing {markdown_file.name} - Markdown length: {len(markdown_content)} characters")

            # Step 2a & 2b: Run classification and extraction concurrently
            agents_start_time = time.time()
            logger.info("🕐 Starting concurrent classification and extraction")

            # Get filename from markdown_file for citation generation
            filename = markdown_file.stem
            
            classification_task = self._classify_document(markdown_content, country)
            extraction_task = self._extract_data(markdown_content, country, filename)

            classification_result, extraction_result = await asyncio.gather(
                classification_task, extraction_task, return_exceptions=True
            )

            agents_time = time.time() - agents_start_time
            logger.info(f"⏱️ Classification and extraction completed in {agents_time:.2f} seconds")

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
            validation_result = {}
            if not isinstance(extraction_result, dict) or "error" not in extraction_result:
                try:
                    # Get the expense type from classification result
                    expense_type = "All"  # Default fallback
                    if isinstance(classification_result, dict) and "expense_type" in classification_result:
                        expense_type = classification_result["expense_type"] or "All"

                    compliance_start_time = time.time()
                    logger.info(f"🕐 Starting compliance analysis for {markdown_file.name} (type: {expense_type})")

                    compliance_analysis_result = await self._analyze_compliance(
                        extraction_result, country, icp, expense_type
                    )

                    compliance_time = time.time() - compliance_start_time
                    logger.info(f"⏱️ Compliance analysis completed in {compliance_time:.2f} seconds")

                    # Handle the new return format (compliance_result, validation_result)
                    if isinstance(compliance_analysis_result, tuple):
                        compliance_result, validation_result = compliance_analysis_result
                    else:
                        compliance_result = compliance_analysis_result
                        validation_result = {}

                    logger.info(f"Compliance analysis completed for {markdown_file.name}")
                except Exception as e:
                    logger.error(f"Compliance analysis failed: {str(e)}")
                    compliance_result = {"error": str(e)}
                    validation_result = {}
            else:
                logger.warning(f"Skipping compliance analysis for {markdown_file.name} due to extraction failure")

            total_document_time = time.time() - document_start_time
            logger.info(f"⏱️ Total document processing time for {markdown_file.name}: {total_document_time:.2f} seconds")

            return {
                "file_name": markdown_file.name,
                "classification": classification_result,
                "extraction": extraction_result,
                "compliance": compliance_result,
                "validation": validation_result or {},
                "processing_time": {
                    "total_seconds": round(total_document_time, 2),
                    "agents_seconds": round(agents_time, 2) if 'agents_time' in locals() else 0,
                    "compliance_seconds": round(compliance_time, 2) if 'compliance_time' in locals() else 0
                },
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
            start_time = time.time()
            logger.debug(f"🕐 Sending markdown to classification agent - Length: {len(markdown_content)}")

            result = classify_file(markdown_content, country)

            classification_time = time.time() - start_time
            logger.debug(f"⏱️ Classification agent completed in {classification_time:.2f} seconds")

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

    async def _extract_data(self, markdown_content: str, country: str, filename: str = None) -> Dict:
        """Extract data using data extraction agent with integrated citation generation."""
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

            start_time = time.time()
            logger.debug(f"🕐 Sending markdown to data extraction agent - Length: {len(markdown_content)}")

            # Step 1: Normal extraction
            result = extract_data_from_receipt(compliance_json, markdown_content)

            extraction_time = time.time() - start_time
            logger.debug(f"⏱️ Data extraction agent completed in {extraction_time:.2f} seconds")

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
            
            # Step 2: Citation generation (if filename provided)
            if filename and isinstance(parsed_result, dict) and "error" not in parsed_result:
                try:
                    citations = generate_citations(
                        structured_output=parsed_result,
                        extraction_requirements=compliance_json,
                        markdown_content=markdown_content,
                        filename=filename
                    )
                    
                    # Log citation statistics
                    citation_stats = get_citation_stats(citations)
                    logger.info(f"Citations for {filename}: {citation_stats.get('fields_with_field_citations', 0)}/{citation_stats.get('total_fields', 0)} field citations, {citation_stats.get('fields_with_value_citations', 0)}/{citation_stats.get('total_fields', 0)} value citations")
                    
                except Exception as e:
                    logger.error(f"Citation generation failed for {filename}: {e}")
            
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

            start_time = time.time()
            logger.debug(f"🕐 Sending extracted data to compliance agent - Keys: {list(extraction_result.keys())}")

            compliance_result = await analyze_compliance_issues(
                country, receipt_type, icp, compliance_data, extraction_result
            )

            compliance_agent_time = time.time() - start_time
            logger.debug(f"⏱️ Compliance agent completed in {compliance_agent_time:.2f} seconds")

            # Handle the new return format (response, validation_results)
            if isinstance(compliance_result, tuple):
                result, validation_results = compliance_result
            else:
                result = compliance_result
                validation_results = None

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
                return parsed_result, validation_results
            else:
                # If result doesn't have content attribute, assume it's already parsed
                parsed_result = result
                logger.debug(f"Compliance analysis result (no content attr): {parsed_result}")
                return parsed_result, validation_results
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in compliance analysis: {str(e)}")
            logger.error(f"Raw content: {getattr(result, 'content', 'No content attribute') if 'result' in locals() else 'No result'}")
            return {"error": f"Invalid JSON response from compliance analysis agent: {str(e)}"}, None
        except Exception as e:
            logger.error(f"Compliance analysis error: {str(e)}")
            return {"error": str(e)}, None

    def _save_single_result(self, result: Dict) -> None:
        """Save a single processing result immediately after processing."""
        if result.get("status") == "completed":
            file_name = result.get("file_name", "unknown")
            # Remove .md extension and add .json
            base_name = pathlib.Path(file_name).stem

            # Create results directory
            results_dir = pathlib.Path("results")
            results_dir.mkdir(exist_ok=True)
            output_file = results_dir / f"{base_name}.json"

            # Create comprehensive result structure (without UQLM validation)
            individual_result = {
                "source_file": file_name,
                "processing_timestamp": datetime.now().isoformat(),
                "dataset_metadata": result.get("dataset_metadata", {}),
                "classification_result": result.get("classification", {}),
                "extraction_result": result.get("extraction", {}),
                "compliance_result": result.get("compliance", {}),
                "processing_status": result.get("status", "unknown"),
                "uqlm_validation_available": bool(result.get("validation", {}))
            }

            try:
                # Convert the entire result to JSON-serializable format first
                serializable_result = self._make_json_serializable(individual_result)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2)
                logger.info(f"💾 Saved individual result: {output_file}")

                # Save detailed validation results if available
                validation_data = result.get("validation", {})
                if validation_data:
                    validation_dir = pathlib.Path("validation_results")
                    validation_dir.mkdir(exist_ok=True)
                    validation_file = validation_dir / f"{base_name}_validation.json"

                    # Create a more readable validation report
                    readable_validation = self._create_readable_validation_report(validation_data)

                    with open(validation_file, 'w', encoding='utf-8') as f:
                        json.dump(readable_validation, f, indent=2)
                    logger.info(f"💾 Saved validation result: {validation_file}")

            except Exception as e:
                logger.error(f"Failed to save result for {file_name}: {e}")
                # Try to save a simplified version without validation data
                try:
                    simplified_result = {
                        "source_file": file_name,
                        "processing_timestamp": datetime.now().isoformat(),
                        "dataset_metadata": result.get("dataset_metadata", {}),
                        "classification_result": result.get("classification", {}),
                        "extraction_result": result.get("extraction", {}),
                        "compliance_result": result.get("compliance", {}),
                        "validation_result": {"error": "Validation data could not be serialized"},
                        "processing_status": result.get("status", "unknown")
                    }
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(simplified_result, f, indent=2)
                    logger.info(f"💾 Saved simplified result: {output_file}")
                except Exception as e2:
                    logger.error(f"Failed to save even simplified result for {file_name}: {e2}")
        else:
            logger.warning(f"Skipping result save for failed processing: {result.get('file_name', 'unknown')}")

    def _save_individual_results(self, results: List[Dict]) -> None:
        """Legacy method - individual results are now saved incrementally during processing."""
        # Results are now saved incrementally via _save_single_result() during processing
        # This method is kept for compatibility but does minimal work
        logger.info(f"Individual results already saved incrementally for {len(results)} files")

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

    def _make_json_serializable(self, obj, _seen=None, _depth=0):
        """Convert objects to JSON-serializable format with recursion protection."""
        if _seen is None:
            _seen = set()

        # Prevent infinite recursion with depth limit
        if _depth > 10:
            return f"<Max depth reached for {type(obj).__name__}>"

        # Check for circular references
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<Circular reference to {type(obj).__name__}>"

        try:
            # Try direct JSON serialization first
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass

        # Add to seen set for recursion protection
        _seen.add(obj_id)

        try:
            if hasattr(obj, '__dict__'):
                # Convert dataclass or object to dict
                result = {}
                for key, value in obj.__dict__.items():
                    try:
                        result[key] = self._make_json_serializable(value, _seen, _depth + 1)
                    except Exception as e:
                        logger.warning(f"Failed to serialize field {key}: {e}")
                        result[key] = str(value)  # Fallback to string representation
                return result
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    try:
                        result[key] = self._make_json_serializable(value, _seen, _depth + 1)
                    except Exception as e:
                        logger.warning(f"Failed to serialize dict key {key}: {e}")
                        result[key] = str(value)  # Fallback to string representation
                return result
            elif isinstance(obj, (list, tuple)):
                result = []
                for item in obj:
                    try:
                        result.append(self._make_json_serializable(item, _seen, _depth + 1))
                    except Exception as e:
                        logger.warning(f"Failed to serialize list item: {e}")
                        result.append(str(item))  # Fallback to string representation
                return result
            elif hasattr(obj, 'value'):
                # Handle enum values
                return obj.value
            else:
                # Fallback to string representation for unknown types
                return str(obj)
        finally:
            # Remove from seen set when done
            _seen.discard(obj_id)



    def _create_readable_validation_report(self, validation_data):
        """Create a readable validation report similar to terminal logs."""
        if not validation_data:
            return {"error": "No validation data available"}

        summary = validation_data.get('validation_summary', {})
        dimensional = validation_data.get('dimensional_analysis', {})
        metadata = validation_data.get('compliance_metadata', {})

        report = {
            "validation_report": {
                "timestamp": datetime.now().isoformat(),
                "overall_assessment": {
                    "confidence_score": summary.get('overall_confidence', 0),
                    "reliability_level": summary.get('reliability_level', 'UNKNOWN'),
                    "is_reliable": summary.get('is_reliable', False),
                    "recommendation": summary.get('recommendation', 'No recommendation available')
                },
                "critical_issues_summary": {
                    "total_issues": len(summary.get('critical_issues', [])),
                    "issues": summary.get('critical_issues', [])
                },
                "dimensional_analysis_summary": {}
            },
            "detailed_analysis": {
                "metadata": {
                    "country": metadata.get('country', 'Unknown'),
                    "receipt_type": metadata.get('receipt_type', 'Unknown'),
                    "icp": metadata.get('icp', 'Unknown'),
                    "validation_method": metadata.get('validation_method', 'UQLM'),
                    "panel_judges": metadata.get('panel_judges', 0),
                    "original_issues_found": metadata.get('original_issues_found', 0)
                },
                "dimension_details": {}
            }
        }

        # Add dimensional analysis in readable format
        for dimension, result in dimensional.items():
            if hasattr(result, '__dict__'):
                dimension_data = {
                    "confidence_score": getattr(result, 'confidence_score', 0),
                    "reliability_level": getattr(result, 'reliability_level', 'unknown'),
                    "summary": getattr(result, 'summary', 'No summary available'),
                    "issues_found": getattr(result, 'issues', []),
                    "total_issues": len(getattr(result, 'issues', []))
                }
            else:
                dimension_data = str(result)

            # Add to summary
            report["validation_report"]["dimensional_analysis_summary"][dimension] = {
                "confidence": dimension_data.get("confidence_score", 0) if isinstance(dimension_data, dict) else "N/A",
                "reliability": dimension_data.get("reliability_level", "unknown") if isinstance(dimension_data, dict) else "N/A",
                "issues_count": dimension_data.get("total_issues", 0) if isinstance(dimension_data, dict) else 0
            }

            # Add to detailed analysis
            report["detailed_analysis"]["dimension_details"][dimension] = dimension_data

        return report
