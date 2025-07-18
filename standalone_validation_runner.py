#!/usr/bin/env python3
"""
Standalone LLM-as-Judge Validation Runner

This script runs LLM-as-judge validation on existing agent outputs and LLM image quality assessment results.
It operates independently from the main workflow pipeline.

Usage:
    python standalone_validation_runner.py --results-dir results --quality-dir llm_quality_reports
"""

import asyncio
import argparse
import json
import logging
import os
import pathlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Import validation classes
from llm_output_checker import ExpenseComplianceUQLMValidator
from llm_quality_validator import ImageQualityUQLMValidator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StandaloneValidationRunner:
    """Standalone runner for LLM-as-judge validation of agent outputs."""
    
    def __init__(self, results_dir: str = "results", quality_dir: str = "llm_quality_reports",
                 validation_output_dir: str = "validation_results"):
        """
        Initialize the validation runner.
        
        Args:
            results_dir: Directory containing agent output JSON files
            quality_dir: Directory containing LLM image quality assessment results
            validation_output_dir: Directory to save validation results
        """
        self.results_dir = pathlib.Path(results_dir)
        self.quality_dir = pathlib.Path(quality_dir)
        self.validation_output_dir = pathlib.Path(validation_output_dir)
        
        # Create output directory
        self.validation_output_dir.mkdir(exist_ok=True)
        
        # Initialize LLM for validation
        self.primary_llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        logger.info(f"Initialized validation runner:")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Quality directory: {self.quality_dir}")
        logger.info(f"  Output directory: {self.validation_output_dir}")
    
    async def run_validation(self, validate_compliance: bool = True, 
                           validate_quality: bool = True) -> Dict[str, Any]:
        """
        Run standalone validation on existing outputs.
        
        Args:
            validate_compliance: Whether to validate compliance analysis results
            validate_quality: Whether to validate LLM image quality assessments
            
        Returns:
            Summary of validation results
        """
        start_time = time.time()
        logger.info("🚀 Starting standalone LLM-as-judge validation")
        
        validation_summary = {
            "validation_timestamp": datetime.now().isoformat(),
            "compliance_validation": {"enabled": validate_compliance, "results": []},
            "quality_validation": {"enabled": validate_quality, "results": []},
            "total_files_processed": 0,
            "total_validation_time": 0
        }
        
        # Validate compliance analysis results
        if validate_compliance and self.results_dir.exists():
            logger.info("🔍 Starting compliance validation...")
            compliance_results = await self._validate_compliance_results()
            validation_summary["compliance_validation"]["results"] = compliance_results
            logger.info(f"✅ Completed compliance validation for {len(compliance_results)} files")
        
        # Validate LLM image quality assessments
        if validate_quality and self.quality_dir.exists():
            logger.info("🔍 Starting quality assessment validation...")
            quality_results = await self._validate_quality_results()
            validation_summary["quality_validation"]["results"] = quality_results
            logger.info(f"✅ Completed quality validation for {len(quality_results)} files")
        
        # Calculate totals
        total_files = (len(validation_summary["compliance_validation"]["results"]) + 
                      len(validation_summary["quality_validation"]["results"]))
        validation_summary["total_files_processed"] = total_files
        validation_summary["total_validation_time"] = round(time.time() - start_time, 2)
        
        # Save summary
        summary_file = self.validation_output_dir / "validation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(validation_summary, f, indent=2)
        
        logger.info(f"🎯 Validation completed in {validation_summary['total_validation_time']} seconds")
        logger.info(f"📊 Total files processed: {total_files}")
        logger.info(f"💾 Summary saved to: {summary_file}")
        
        return validation_summary
    
    async def _validate_compliance_results(self) -> List[Dict[str, Any]]:
        """Validate compliance analysis results from agent outputs."""
        compliance_results = []
        
        # Find all result files
        result_files = list(self.results_dir.glob("*.json"))
        logger.info(f"Found {len(result_files)} result files for compliance validation")
        
        # Initialize compliance validator
        validator = ExpenseComplianceUQLMValidator(self.primary_llm, logger)
        
        for result_file in result_files:
            try:
                logger.info(f"🔍 Validating compliance for: {result_file.name}")
                
                # Load the result file
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # Extract necessary data for validation
                compliance_result = result_data.get("compliance_result", {})
                extraction_result = result_data.get("extraction_result", {})
                
                # Skip if no compliance data
                if not compliance_result or "error" in compliance_result:
                    logger.warning(f"⚠️ Skipping {result_file.name} - no valid compliance data")
                    continue
                
                # Extract metadata
                country = self._extract_country_from_result(result_data)
                icp = self._extract_icp_from_result(result_data)
                receipt_type = self._extract_receipt_type_from_result(result_data)
                
                # Run validation
                validation_start_time = time.time()
                validation_result = await validator.validate_compliance_response(
                    ai_response=json.dumps(compliance_result),
                    country=country,
                    icp=icp,
                    receipt_type=receipt_type,
                    compliance_json=compliance_result,
                    extracted_json=extraction_result
                )
                
                validation_time = time.time() - validation_start_time
                
                # Save individual validation result (transform to expected format)
                validation_file = self.validation_output_dir / f"{result_file.stem}_compliance_validation.json"

                # Transform raw UQLM output to expected format (same as integrated workflow)
                readable_validation = self._create_readable_validation_report(validation_result)
                serializable_result = self._make_json_serializable(readable_validation)

                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2)
                
                # Add to summary (extract from original UQLM format)
                overall_assessment = validation_result.get("validation_report", {}).get("overall_assessment", {})
                compliance_results.append({
                    "source_file": result_file.name,
                    "validation_file": validation_file.name,
                    "validation_time": round(validation_time, 2),
                    "confidence_score": overall_assessment.get("confidence_score", 0),
                    "reliability_level": overall_assessment.get("reliability_level", "unknown"),
                    "is_reliable": overall_assessment.get("is_reliable", False),
                    "status": "completed"
                })
                
                logger.info(f"✅ Compliance validation completed for {result_file.name} in {validation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Error validating compliance for {result_file.name}: {str(e)}")
                compliance_results.append({
                    "source_file": result_file.name,
                    "validation_file": None,
                    "validation_time": 0,
                    "error": str(e),
                    "status": "failed"
                })
        
        return compliance_results

    async def _validate_quality_results(self) -> List[Dict[str, Any]]:
        """Validate LLM image quality assessment results."""
        quality_results = []

        # Find all quality assessment files
        quality_files = list(self.quality_dir.glob("*.json"))
        logger.info(f"Found {len(quality_files)} quality files for validation")

        # Initialize quality validator
        validator = ImageQualityUQLMValidator(self.primary_llm, logger)

        for quality_file in quality_files:
            try:
                logger.info(f"🔍 Validating quality assessment for: {quality_file.name}")

                # Load the quality assessment file
                with open(quality_file, 'r', encoding='utf-8') as f:
                    quality_data = json.load(f)

                # Extract LLM assessment data
                llm_assessment = quality_data.get("llm_quality_result", {})
                if not llm_assessment or "error" in llm_assessment:
                    logger.warning(f"⚠️ Skipping {quality_file.name} - no valid LLM assessment data")
                    continue

                # Find corresponding image file
                image_path = self._find_corresponding_image(quality_file.name)
                if not image_path:
                    logger.warning(f"⚠️ Skipping {quality_file.name} - corresponding image not found")
                    continue

                # Get OpenCV assessment if available
                opencv_assessment = quality_data.get("opencv_quality_result")

                # Run validation
                validation_start_time = time.time()
                validation_result = await validator.validate_quality_assessment(
                    llm_assessment=llm_assessment,
                    image_path=str(image_path),
                    opencv_assessment=opencv_assessment
                )

                validation_time = time.time() - validation_start_time

                # Save individual validation result (transform to expected format)
                validation_file = self.validation_output_dir / f"{quality_file.stem}_quality_validation.json"

                # Transform raw UQLM output to expected format
                readable_validation = self._create_readable_quality_validation_report(validation_result)
                serializable_result = self._make_json_serializable(readable_validation)

                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2)

                # Add to summary (extract from original UQLM format)
                overall_assessment = validation_result.get("validation_report", {}).get("overall_assessment", {})
                quality_results.append({
                    "source_file": quality_file.name,
                    "validation_file": validation_file.name,
                    "validation_time": round(validation_time, 2),
                    "confidence_score": overall_assessment.get("confidence_score", 0),
                    "reliability_level": overall_assessment.get("reliability_level", "unknown"),
                    "is_reliable": overall_assessment.get("is_reliable", False),
                    "status": "completed"
                })

                logger.info(f"✅ Quality validation completed for {quality_file.name} in {validation_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Error validating quality for {quality_file.name}: {str(e)}")
                quality_results.append({
                    "source_file": quality_file.name,
                    "validation_file": None,
                    "validation_time": 0,
                    "error": str(e),
                    "status": "failed"
                })

        return quality_results

    def _extract_country_from_result(self, result_data: Dict) -> str:
        """Extract country from result data."""
        # Try different possible locations for country info
        dataset_metadata = result_data.get("dataset_metadata", {})
        if "country" in dataset_metadata:
            return dataset_metadata["country"]

        classification = result_data.get("classification_result", {})
        if "country" in classification:
            return classification["country"]

        # Default fallback
        return "Unknown"

    def _extract_icp_from_result(self, result_data: Dict) -> str:
        """Extract ICP from result data."""
        dataset_metadata = result_data.get("dataset_metadata", {})
        if "icp" in dataset_metadata:
            return dataset_metadata["icp"]

        # Default fallback
        return "Unknown"

    def _extract_receipt_type_from_result(self, result_data: Dict) -> str:
        """Extract receipt type from result data."""
        classification = result_data.get("classification_result", {})
        if "expense_type" in classification:
            return classification["expense_type"]

        dataset_metadata = result_data.get("dataset_metadata", {})
        if "receipt_type" in dataset_metadata:
            return dataset_metadata["receipt_type"]

        # Default fallback
        return "All"

    def _find_corresponding_image(self, quality_filename: str) -> Optional[pathlib.Path]:
        """Find the corresponding image file for a quality assessment."""
        # Remove quality assessment suffix and extension
        base_name = quality_filename.replace("_quality_assessment.json", "")

        # Common image extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"]

        # Check in expense_files directory
        expense_files_dir = pathlib.Path("expense_files")
        if expense_files_dir.exists():
            for ext in image_extensions:
                image_path = expense_files_dir / f"{base_name}{ext}"
                if image_path.exists():
                    return image_path

        # Check in current directory
        for ext in image_extensions:
            image_path = pathlib.Path(f"{base_name}{ext}")
            if image_path.exists():
                return image_path

        return None

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _create_readable_validation_report(self, validation_data):
        """Create a readable validation report similar to terminal logs and integrated workflow format."""
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

    def _create_readable_quality_validation_report(self, validation_data):
        """Create a readable quality validation report similar to compliance format."""
        if not validation_data:
            return {"error": "No validation data available"}

        summary = validation_data.get('validation_summary', {})
        dimensional = validation_data.get('dimensional_analysis', {})
        judge_assessments = validation_data.get('judge_assessments', {})

        report = {
            "validation_report": {
                "timestamp": datetime.now().isoformat(),
                "validation_type": "image_quality_assessment",
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
                "dimensional_analysis_summary": {},
                "judge_assessments_summary": {}
            },
            "detailed_analysis": {
                "judge_assessments": judge_assessments,
                "dimension_details": {}
            }
        }

        # Add dimensional analysis in readable format
        for dimension, result in dimensional.items():
            if hasattr(result, '__dict__'):
                # Handle dataclass objects
                dimension_data = {
                    "confidence_score": getattr(result, 'confidence_score', 0),
                    "reliability_level": getattr(result, 'reliability_level', 'unknown'),
                    "summary": getattr(result, 'summary', 'No summary available'),
                    "issues_found": getattr(result, 'issues', []),
                    "total_issues": len(getattr(result, 'issues', []))
                }
            elif isinstance(result, dict):
                # Handle regular dictionary objects (quality validation format)
                # Check if it contains the expected structure
                if 'confidence_score' in result and 'reliability_level' in result:
                    dimension_data = {
                        "confidence_score": result.get('confidence_score', 0),
                        "reliability_level": result.get('reliability_level', 'unknown'),
                        "summary": result.get('summary', 'No summary available'),
                        "issues_found": result.get('issues', []),
                        "total_issues": len(result.get('issues', []))
                    }
                else:
                    # The dict might contain a string value that needs parsing
                    # Check if there's a single string value that contains the actual data
                    if len(result) == 1:
                        string_value = list(result.values())[0]
                        if isinstance(string_value, str):
                            try:
                                import ast
                                parsed_result = ast.literal_eval(string_value)
                                if isinstance(parsed_result, dict) and 'confidence_score' in parsed_result:
                                    dimension_data = {
                                        "confidence_score": parsed_result.get('confidence_score', 0),
                                        "reliability_level": parsed_result.get('reliability_level', 'unknown'),
                                        "summary": parsed_result.get('summary', 'No summary available'),
                                        "issues_found": parsed_result.get('issues', []),
                                        "total_issues": len(parsed_result.get('issues', []))
                                    }
                                else:
                                    dimension_data = str(result)
                            except:
                                try:
                                    parsed_result = eval(string_value)
                                    if isinstance(parsed_result, dict) and 'confidence_score' in parsed_result:
                                        dimension_data = {
                                            "confidence_score": parsed_result.get('confidence_score', 0),
                                            "reliability_level": parsed_result.get('reliability_level', 'unknown'),
                                            "summary": parsed_result.get('summary', 'No summary available'),
                                            "issues_found": parsed_result.get('issues', []),
                                            "total_issues": len(parsed_result.get('issues', []))
                                        }
                                    else:
                                        dimension_data = str(result)
                                except:
                                    dimension_data = str(result)
                        else:
                            dimension_data = str(result)
                    else:
                        dimension_data = str(result)
            elif isinstance(result, str):
                # Handle string representations of dictionaries (quality validation format)
                try:
                    # Try ast.literal_eval first
                    import ast
                    parsed_result = ast.literal_eval(result)
                    if isinstance(parsed_result, dict):
                        dimension_data = {
                            "confidence_score": parsed_result.get('confidence_score', 0),
                            "reliability_level": parsed_result.get('reliability_level', 'unknown'),
                            "summary": parsed_result.get('summary', 'No summary available'),
                            "issues_found": parsed_result.get('issues', []),
                            "total_issues": len(parsed_result.get('issues', []))
                        }
                    else:
                        dimension_data = str(result)
                except (ValueError, SyntaxError):
                    # If ast.literal_eval fails, try eval as fallback (less safe but works with single quotes)
                    try:
                        parsed_result = eval(result)
                        if isinstance(parsed_result, dict):
                            dimension_data = {
                                "confidence_score": parsed_result.get('confidence_score', 0),
                                "reliability_level": parsed_result.get('reliability_level', 'unknown'),
                                "summary": parsed_result.get('summary', 'No summary available'),
                                "issues_found": parsed_result.get('issues', []),
                                "total_issues": len(parsed_result.get('issues', []))
                            }
                        else:
                            dimension_data = str(result)
                    except:
                        dimension_data = str(result)
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

        # Process judge assessments summary
        for judge_name, assessment in judge_assessments.items():
            if isinstance(assessment, dict):
                report["validation_report"]["judge_assessments_summary"][judge_name] = {
                    "overall_score": assessment.get("overall_quality_score", "N/A"),
                    "suitable_for_extraction": assessment.get("suitable_for_extraction", False),
                    "status": "error" if "error" in assessment else "completed"
                }

        return report


async def main():
    """Main entry point for standalone validation runner."""
    parser = argparse.ArgumentParser(
        description="Standalone LLM-as-Judge Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate both compliance and quality assessments
  python standalone_validation_runner.py

  # Validate only compliance analysis results
  python standalone_validation_runner.py --no-quality

  # Validate only quality assessments
  python standalone_validation_runner.py --no-compliance

  # Use custom directories
  python standalone_validation_runner.py --results-dir my_results --quality-dir my_quality
        """
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing agent output JSON files (default: results)"
    )

    parser.add_argument(
        "--quality-dir",
        default="llm_quality_reports",
        help="Directory containing LLM image quality assessment results (default: llm_quality_reports)"
    )

    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Directory to save validation results (default: validation_results)"
    )

    parser.add_argument(
        "--no-compliance",
        action="store_true",
        help="Skip compliance validation"
    )

    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality assessment validation"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("❌ ANTHROPIC_API_KEY environment variable is required")
        return 1

    # Check if at least one validation type is enabled
    validate_compliance = not args.no_compliance
    validate_quality = not args.no_quality

    if not validate_compliance and not validate_quality:
        logger.error("❌ At least one validation type must be enabled")
        return 1

    # Check if input directories exist
    if validate_compliance and not pathlib.Path(args.results_dir).exists():
        logger.error(f"❌ Results directory not found: {args.results_dir}")
        return 1

    if validate_quality and not pathlib.Path(args.quality_dir).exists():
        logger.error(f"❌ Quality directory not found: {args.quality_dir}")
        return 1

    try:
        # Initialize and run validation
        runner = StandaloneValidationRunner(
            results_dir=args.results_dir,
            quality_dir=args.quality_dir,
            validation_output_dir=args.output_dir
        )

        summary = await runner.run_validation(
            validate_compliance=validate_compliance,
            validate_quality=validate_quality
        )

        # Print summary
        logger.info("🎯 Validation Summary:")
        logger.info(f"   Total files processed: {summary['total_files_processed']}")
        logger.info(f"   Total validation time: {summary['total_validation_time']} seconds")

        if validate_compliance:
            compliance_count = len(summary['compliance_validation']['results'])
            logger.info(f"   Compliance validations: {compliance_count}")

        if validate_quality:
            quality_count = len(summary['quality_validation']['results'])
            logger.info(f"   Quality validations: {quality_count}")

        logger.info(f"   Results saved to: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
