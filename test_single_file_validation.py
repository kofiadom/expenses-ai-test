#!/usr/bin/env python3
"""
Single File Validation Test Script

This script runs both compliance validation and LLM image quality assessment validation
on a single file to demonstrate the new format transformation working correctly.

Usage:
    python test_single_file_validation.py --file austrian_file
    python test_single_file_validation.py --file german_file_2 --skip-quality
"""

import asyncio
import argparse
import json
import os
import pathlib
import time
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Import validation classes
from llm_output_checker import ExpenseComplianceUQLMValidator
from llm_quality_validator import ImageQualityUQLMValidator

# Load environment variables
load_dotenv()

class SingleFileValidationTester:
    """Test validation on a single file to demonstrate the new format."""
    
    def __init__(self, results_dir: str = "results", quality_dir: str = "llm_quality_reports"):
        """Initialize the tester."""
        self.results_dir = pathlib.Path(results_dir)
        self.quality_dir = pathlib.Path(quality_dir)
        self.output_dir = pathlib.Path("test_validation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize LLM for validation
        self.primary_llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        print(f"🔧 Initialized Single File Validation Tester")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Quality directory: {self.quality_dir}")
        print(f"   Output directory: {self.output_dir}")
    
    async def test_single_file(self, file_name: str, test_compliance: bool = True, 
                              test_quality: bool = True) -> Dict[str, Any]:
        """Test validation on a single file."""
        
        print(f"\n🧪 Testing Validation on Single File: {file_name}")
        print("=" * 60)
        
        results = {
            "file_name": file_name,
            "compliance_validation": {"enabled": test_compliance, "status": "skipped"},
            "quality_validation": {"enabled": test_quality, "status": "skipped"},
            "total_time": 0
        }
        
        start_time = time.time()
        
        # Test compliance validation
        if test_compliance:
            print(f"\n📋 Testing Compliance Validation")
            compliance_result = await self._test_compliance_validation(file_name)
            results["compliance_validation"] = compliance_result
        
        # Test quality validation  
        if test_quality:
            print(f"\n🖼️ Testing LLM Image Quality Assessment Validation")
            quality_result = await self._test_quality_validation(file_name)
            results["quality_validation"] = quality_result
        
        results["total_time"] = round(time.time() - start_time, 2)
        
        # Save test summary
        summary_file = self.output_dir / f"{file_name}_test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Test Summary:")
        print(f"   Total time: {results['total_time']} seconds")
        print(f"   Compliance validation: {results['compliance_validation']['status']}")
        print(f"   Quality validation: {results['quality_validation']['status']}")
        print(f"   Summary saved: {summary_file}")
        
        return results
    
    async def _test_compliance_validation(self, file_name: str) -> Dict[str, Any]:
        """Test compliance validation on a single file."""
        
        result = {"enabled": True, "status": "failed", "error": None, "validation_time": 0}
        
        try:
            # Find the result file
            result_file = self.results_dir / f"{file_name}.json"
            if not result_file.exists():
                raise FileNotFoundError(f"Result file not found: {result_file}")
            
            print(f"   📁 Loading result file: {result_file.name}")
            
            # Load result data
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Extract required data
            compliance_result = result_data.get("compliance_result", {})
            extraction_result = result_data.get("extraction_result", {})
            
            if not compliance_result or not extraction_result:
                raise ValueError("Missing compliance or extraction data in result file")
            
            # Extract metadata
            country = self._extract_country_from_result(result_data)
            icp = self._extract_icp_from_result(result_data)
            receipt_type = self._extract_receipt_type_from_result(result_data)
            
            print(f"   📊 Metadata: {country} | {receipt_type} | {icp}")
            
            # Initialize validator
            validator = ExpenseComplianceUQLMValidator(
                primary_llm=self.primary_llm
            )
            
            # Run validation
            validation_start_time = time.time()
            print(f"   🕐 Running UQLM compliance validation...")
            
            validation_result = await validator.validate_compliance_response(
                ai_response=json.dumps(compliance_result),
                country=country,
                icp=icp,
                receipt_type=receipt_type,
                compliance_json=compliance_result,
                extracted_json=extraction_result
            )
            
            validation_time = time.time() - validation_start_time
            
            # Transform to expected format
            print(f"   🔄 Transforming to expected format...")
            readable_validation = self._create_readable_validation_report(validation_result)
            
            # Save validation result
            validation_file = self.output_dir / f"{file_name}_compliance_validation.json"
            serializable_result = self._make_json_serializable(readable_validation)
            
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2)
            
            # Extract summary info
            overall_assessment = readable_validation.get("validation_report", {}).get("overall_assessment", {})
            
            result.update({
                "status": "completed",
                "validation_time": round(validation_time, 2),
                "validation_file": validation_file.name,
                "confidence_score": overall_assessment.get("confidence_score", 0),
                "reliability_level": overall_assessment.get("reliability_level", "unknown"),
                "is_reliable": overall_assessment.get("is_reliable", False)
            })
            
            print(f"   ✅ Compliance validation completed in {validation_time:.2f}s")
            print(f"   📊 Confidence: {result['confidence_score']:.3f} | Reliability: {result['reliability_level']}")
            print(f"   💾 Saved: {validation_file.name}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"   ❌ Compliance validation failed: {str(e)}")
        
        return result
    
    async def _test_quality_validation(self, file_name: str) -> Dict[str, Any]:
        """Test quality validation on a single file."""
        
        result = {"enabled": True, "status": "failed", "error": None, "validation_time": 0}
        
        try:
            # Find the quality assessment file
            quality_file = self.quality_dir / f"llm_quality_{file_name}.json"
            if not quality_file.exists():
                raise FileNotFoundError(f"Quality assessment file not found: {quality_file}")
            
            print(f"   📁 Loading quality file: {quality_file.name}")
            
            # Load quality data
            with open(quality_file, 'r', encoding='utf-8') as f:
                quality_data = json.load(f)
            
            # Find corresponding image
            image_path = self._find_image_for_file(file_name)
            if not image_path:
                raise FileNotFoundError(f"No image found for {file_name}")
            
            print(f"   🖼️ Found image: {image_path.name}")
            
            # Initialize validator
            validator = ImageQualityUQLMValidator(
                primary_llm=self.primary_llm
            )
            
            # Run validation
            validation_start_time = time.time()
            print(f"   🕐 Running UQLM quality validation...")
            
            validation_result = await validator.validate_quality_assessment(
                llm_assessment=quality_data,
                image_path=str(image_path)
            )
            
            validation_time = time.time() - validation_start_time
            
            # Transform to expected format
            print(f"   🔄 Transforming to expected format...")
            readable_validation = self._create_readable_quality_validation_report(validation_result)
            
            # Save validation result
            validation_file = self.output_dir / f"{file_name}_quality_validation.json"
            serializable_result = self._make_json_serializable(readable_validation)
            
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2)
            
            # Extract summary info
            overall_assessment = readable_validation.get("validation_report", {}).get("overall_assessment", {})
            
            result.update({
                "status": "completed",
                "validation_time": round(validation_time, 2),
                "validation_file": validation_file.name,
                "confidence_score": overall_assessment.get("confidence_score", 0),
                "reliability_level": overall_assessment.get("reliability_level", "unknown"),
                "is_reliable": overall_assessment.get("is_reliable", False)
            })
            
            print(f"   ✅ Quality validation completed in {validation_time:.2f}s")
            print(f"   📊 Confidence: {result['confidence_score']:.3f} | Reliability: {result['reliability_level']}")
            print(f"   💾 Saved: {validation_file.name}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"   ❌ Quality validation failed: {str(e)}")
        
        return result
    
    def _extract_country_from_result(self, result_data: Dict) -> str:
        """Extract country from result data."""
        metadata = result_data.get("dataset_metadata", {})
        return metadata.get("country", "Unknown")
    
    def _extract_icp_from_result(self, result_data: Dict) -> str:
        """Extract ICP from result data."""
        metadata = result_data.get("dataset_metadata", {})
        return metadata.get("icp", "Unknown")
    
    def _extract_receipt_type_from_result(self, result_data: Dict) -> str:
        """Extract receipt type from result data."""
        classification = result_data.get("classification_result", {})
        return classification.get("receipt_type", "Unknown")
    
    def _find_image_for_file(self, file_name: str) -> Optional[pathlib.Path]:
        """Find the corresponding image file."""
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.pdf']

        # Check multiple directories
        search_dirs = ["expense_files", "dataset", "images"]

        for dir_name in search_dirs:
            search_dir = pathlib.Path(dir_name)
            if search_dir.exists():
                for ext in extensions:
                    image_path = search_dir / f"{file_name}{ext}"
                    if image_path.exists():
                        return image_path

        return None
    
    # Include the transformation methods from standalone_validation_runner.py
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
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
        """Create a readable validation report similar to integrated workflow format."""
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
    """Main function."""
    parser = argparse.ArgumentParser(description="Test validation on a single file")
    parser.add_argument("--file", required=True, help="File name to test (without extension)")
    parser.add_argument("--skip-compliance", action="store_true", help="Skip compliance validation")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality validation")
    
    args = parser.parse_args()
    
    try:
        tester = SingleFileValidationTester()
        
        await tester.test_single_file(
            file_name=args.file,
            test_compliance=not args.skip_compliance,
            test_quality=not args.skip_quality
        )
        
        print(f"\n🎉 Single file validation test completed!")
        print(f"📁 Check 'test_validation_output' directory for results")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
