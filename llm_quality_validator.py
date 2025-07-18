import os
import json
import re
import logging
import time
import base64
from langchain_anthropic import ChatAnthropic
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from agno.utils.log import logger

# Import uqlm LLMPanel - based on the working example
try:
    from uqlm import LLMPanel
    UQLM_AVAILABLE = True
except ImportError:
    print("Warning: UQLM library not available. Install with: pip install uqlm")
    LLMPanel = None
    UQLM_AVAILABLE = False


class QualityValidationDimension(Enum):
    VISUAL_ACCURACY = "visual_accuracy"
    QUANTITATIVE_RELIABILITY = "quantitative_reliability"
    SEVERITY_ASSESSMENT = "severity_assessment"
    RECOMMENDATION_VALIDITY = "recommendation_validity"
    CONSISTENCY_CHECK = "consistency_check"
    HALLUCINATION_DETECTION = "hallucination_detection"


@dataclass
class QualityValidationResult:
    dimension: QualityValidationDimension
    confidence_score: float  # 0.0 to 1.0
    issues: List[str]
    summary: str
    raw_response: str
    reliability_level: str  # 'high', 'medium', 'low'

    def to_dict(self) -> Dict[str, Any]:
        """Convert QualityValidationResult to dictionary for JSON serialization"""
        return {
            "dimension": self.dimension.value,
            "confidence_score": self.confidence_score,
            "issues": self.issues,
            "summary": self.summary,
            "raw_response": self.raw_response,
            "reliability_level": self.reliability_level
        }


class ImageQualityUQLMValidator:
    """
    UQLM-based validator for LLM image quality assessment results.
    Validates LLM quality assessments against actual images using LLM-as-a-Judge approach.
    """

    def __init__(self, primary_llm, logger: logging.Logger = None):
        """
        Initialize UQLM quality validator with judge panel.
        
        Args:
            primary_llm: The primary LLM instance
            logger: Logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.primary_llm = primary_llm

        self.llm1 = ChatAnthropic(model="claude-opus-4-20250514",
                        api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.llm2 = ChatAnthropic(model="claude-sonnet-4-20250514",
                        api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Create judge panel with multiple instances (following uqlm pattern)
        self.judge_llms = [self.llm1, self.llm2]

        if not UQLM_AVAILABLE:
            self.logger.warning("⚠️ UQLM library not available. Validation will be skipped.")
            self.panel = None
            return

        try:
            self.panel = LLMPanel(llm=primary_llm, judges=self.judge_llms)
            self.logger.info("🎯 UQLM LLM Panel initialized successfully with 3 judges for quality validation")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize UQLM panel: {str(e)}")
            self.panel = None

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for validation prompts"""
        try:
            # Handle format conversion if needed
            with Image.open(image_path) as img:
                if img.format and img.format.lower() not in ['jpeg', 'jpg', 'png', 'webp', 'gif']:
                    # Convert to RGB and save as JPEG
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    import io
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr = img_byte_arr.getvalue()
                    return base64.b64encode(img_byte_arr).decode('utf-8')
                else:
                    # Use original file
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8')
                        
        except Exception as e:
            raise Exception(f"Error encoding image {image_path}: {str(e)}")

    async def validate_quality_assessment(self,
                                        llm_assessment: Dict,
                                        image_path: str,
                                        opencv_assessment: Dict = None) -> Dict[str, Any]:
        """
        Validate LLM's quality assessment response using UQLM panel and get judge assessments

        Args:
            llm_assessment: The LLM's quality assessment results (Dict)
            image_path: Path to the original image file
            opencv_assessment: Optional OpenCV assessment for comparison

        Returns:
            Comprehensive validation results with UQLM uncertainty quantification and judge assessments
        """
        if not self.panel:
            raise ValueError("UQLM panel not properly initialized")

        validation_start_time = time.time()
        self.logger.info("🕐 Starting UQLM-based quality assessment validation with judge assessments")

        # Encode image for validation
        try:
            base64_image = self.encode_image(image_path)
            self.logger.info("✅ Successfully encoded image for validation")
        except Exception as e:
            self.logger.error(f"❌ Failed to encode image: {str(e)}")
            return self._create_error_validation_result(f"Image encoding failed: {str(e)}")

        # Step 1: Get independent quality assessments from each judge
        self.logger.info("🤖 Step 1: Getting independent quality assessments from judge LLMs")
        judge_assessments = await self._get_judge_quality_assessments(base64_image, image_path)

        # Step 2: Validate each dimension using UQLM panel
        self.logger.info("🎯 Step 2: Performing UQLM validation of original assessment")
        validation_results = {}

        for dimension in QualityValidationDimension:
            self.logger.info(f"🔍 Validating {dimension.value} using UQLM panel")

            try:
                # Create dimension-specific validation prompt
                validation_prompt = self._create_validation_prompt(
                    llm_assessment, base64_image, image_path, opencv_assessment, dimension
                )

                # Use UQLM panel for validation
                dimension_result = await self._validate_dimension_with_panel(
                    validation_prompt, dimension
                )
                validation_results[dimension.value] = dimension_result.to_dict()

            except Exception as e:
                self.logger.error(f"❌ Error validating {dimension.value}: {str(e)}")
                validation_results[dimension.value] = self._create_error_result(dimension, str(e)).to_dict()

        # Step 3: Format comprehensive results including judge assessments
        formatted_res = self._format_validation_results(validation_results, llm_assessment, image_path, judge_assessments)

        validation_time = time.time() - validation_start_time
        self.logger.info(f"⏱️ UQLM quality validation completed in {validation_time:.2f} seconds")
        self.logger.info(f"Quality validation results: {formatted_res['validation_summary']}")
        return formatted_res

    def _create_validation_prompt(self, llm_assessment: Dict, base64_image: str, 
                                image_path: str, opencv_assessment: Dict, 
                                dimension: QualityValidationDimension) -> str:
        """Create dimension-specific validation prompt for quality assessment"""
        
        dimension_instructions = {
            QualityValidationDimension.VISUAL_ACCURACY: {
                "focus": "Visual Accuracy Analysis",
                "description": "Verify that the LLM correctly identified visual quality issues present in the image.",
                "checks": [
                    "Are all quality issues claimed by the LLM actually visible in the image?",
                    "Did the LLM miss any obvious quality problems?",
                    "Are the detected issues accurately described?",
                    "Does the visual evidence support the LLM's findings?"
                ]
            },
            QualityValidationDimension.QUANTITATIVE_RELIABILITY: {
                "focus": "Quantitative Reliability",
                "description": "Assess whether quantitative measures (0.0-1.0 scales) are reasonable and consistent.",
                "checks": [
                    "Are blur intensity measures (0.0-1.0) appropriate for the visible blur level?",
                    "Do damage percentages align with what's visible in the image?",
                    "Are confidence scores reasonable for the assessment quality?",
                    "Do quantitative measures correlate logically with detected issues?"
                ]
            },
            QualityValidationDimension.SEVERITY_ASSESSMENT: {
                "focus": "Severity Assessment",
                "description": "Validate that severity levels (low/medium/high/critical) match the actual impact on extraction.",
                "checks": [
                    "Are 'critical' severity issues truly extraction-blocking?",
                    "Are 'low' severity issues appropriately minor?",
                    "Do severity levels match the quantitative measures?",
                    "Are severity assessments consistent across similar issues?"
                ]
            },
            QualityValidationDimension.RECOMMENDATION_VALIDITY: {
                "focus": "Recommendation Validity",
                "description": "Assess whether recommendations are practical and appropriate for the identified issues.",
                "checks": [
                    "Are recommendations specific and actionable?",
                    "Do recommendations address the actual problems identified?",
                    "Are the suggested actions appropriate for the issue severity?",
                    "Are any recommendations missing or inappropriate?"
                ]
            },
            QualityValidationDimension.CONSISTENCY_CHECK: {
                "focus": "Internal Consistency",
                "description": "Check for logical consistency within the LLM's assessment.",
                "checks": [
                    "Do confidence scores align with the certainty of descriptions?",
                    "Are severity levels consistent with quantitative measures?",
                    "Does the overall suitability judgment match individual issue assessments?",
                    "Are there any contradictory findings within the assessment?"
                ]
            },
            QualityValidationDimension.HALLUCINATION_DETECTION: {
                "focus": "Hallucination Detection",
                "description": "Identify any fabricated quality issues or invented problems not visible in the image.",
                "checks": [
                    "Are there any claimed issues not actually present in the image?",
                    "Are all described locations and areas actually visible?",
                    "Are quantitative measures realistic rather than fabricated?",
                    "Are there any impossible or contradictory quality claims?"
                ]
            }
        }
        
        dim_info = dimension_instructions[dimension]
        checks_text = "\n".join([f"- {check}" for check in dim_info["checks"]])
        
        # Include OpenCV comparison if available
        opencv_context = ""
        if opencv_assessment:
            opencv_context = f"""
            
            OPENCV ASSESSMENT FOR COMPARISON:
            Score: {opencv_assessment.get('quality_score', 'N/A')}/100
            Level: {opencv_assessment.get('quality_level', 'N/A')}
            Passed: {opencv_assessment.get('quality_passed', 'N/A')}
            """

        return f"""You are an expert image quality validation specialist. Your task is to validate an LLM's image quality assessment by examining the actual image and the LLM's findings.

                VALIDATION FOCUS: {dim_info["focus"]}
                {dim_info["description"]}

                IMAGE TO ANALYZE:
                [Base64 image data: {base64_image[:100]}...]
                Image path: {image_path}

                LLM QUALITY ASSESSMENT TO VALIDATE:
                {json.dumps(llm_assessment, indent=2)}
                {opencv_context}

                VALIDATION CHECKLIST:
                {checks_text}

                INSTRUCTIONS:
                1. Carefully examine the provided image
                2. Review the LLM's quality assessment findings
                3. Validate each aspect according to the checklist above
                4. Identify any issues, inconsistencies, or inaccuracies
                5. Provide a confidence score for the {dimension.value} dimension

                REQUIRED OUTPUT FORMAT:
                ```json
                {{
                "issues": ["list of specific validation issues found"],
                "summary": "detailed analysis of the {dimension.value} dimension",
                "confidence_score": 0.85,
                "reliability_level": "high|medium|low"
                }}
                ```

                At the end, explicitly state: "confidence_score: [your numerical score]"
                """

    async def _get_judge_quality_assessments(self, base64_image: str, image_path: str) -> Dict[str, Dict]:
        """Get independent quality assessments from each judge LLM"""
        judge_assessments = {}

        # Create quality assessment prompt (similar to original LLM assessment)
        assessment_prompt = self._create_judge_assessment_prompt(base64_image, image_path)

        for i, judge_llm in enumerate(self.judge_llms, 1):
            judge_name = f"judge_{i}"
            self.logger.info(f"🤖 Getting quality assessment from {judge_name}")

            try:
                judge_start_time = time.time()

                # Get assessment from judge LLM
                response = await judge_llm.ainvoke(assessment_prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Parse the judge's assessment
                judge_assessment = self._parse_judge_assessment(response_content, judge_name)
                judge_assessments[judge_name] = judge_assessment

                judge_time = time.time() - judge_start_time
                self.logger.info(f"✅ {judge_name} assessment completed in {judge_time:.2f}s - Score: {judge_assessment.get('overall_quality_score', 'N/A')}/10")

            except Exception as e:
                self.logger.error(f"❌ Error getting assessment from {judge_name}: {str(e)}")
                judge_assessments[judge_name] = {
                    "error": f"Assessment failed: {str(e)}",
                    "judge_name": judge_name,
                    "overall_quality_score": 0,
                    "suitable_for_extraction": False
                }

        return judge_assessments

    def _create_judge_assessment_prompt(self, base64_image: str, image_path: str) -> str:
        """Create prompt for judge LLMs to provide their own quality assessment"""
        return f"""You are an expert image quality assessment specialist. Analyze this receipt/invoice image and provide a comprehensive quality assessment.

IMAGE TO ANALYZE:
[Base64 image data: {base64_image[:100]}...]
Image path: {image_path}

ASSESSMENT INSTRUCTIONS:
Analyze the image for the following quality issues and provide detailed assessments:

1. **Blur Detection**: Assess image sharpness and text clarity
2. **Contrast Assessment**: Evaluate text-to-background contrast
3. **Glare Identification**: Check for reflections or bright spots
4. **Water Stains**: Look for water damage or discoloration
5. **Tears or Folds**: Identify physical damage or creases
6. **Cut-off Detection**: Check if any parts of the document are missing
7. **Missing Sections**: Verify all expected receipt sections are present
8. **Obstructions**: Identify anything blocking important information

For each issue, provide:
- detected: true/false
- severity_level: "low", "medium", "high", or "critical"
- confidence_score: 0.0 to 1.0 (your confidence in this assessment)
- quantitative_measure: 0.0 to 1.0 (objective measure of the issue intensity)
- description: One sentence describing what you observe
- recommendation: Practical suggestion for improvement (or "No action needed")

REQUIRED OUTPUT FORMAT:
```json
{{
  "blur_detection": {{
    "detected": true/false,
    "severity_level": "low|medium|high|critical",
    "confidence_score": 0.0-1.0,
    "quantitative_measure": 0.0-1.0,
    "description": "One sentence description",
    "recommendation": "Practical recommendation"
  }},
  "contrast_assessment": {{ ... }},
  "glare_identification": {{ ... }},
  "water_stains": {{ ... }},
  "tears_or_folds": {{ ... }},
  "cut_off_detection": {{ ... }},
  "missing_sections": {{ ... }},
  "obstructions": {{ ... }},
  "overall_quality_score": 1-10,
  "suitable_for_extraction": true/false,
  "assessment_method": "LLM_Judge",
  "model_used": "judge_llm"
}}
```

Provide only the JSON response, no additional text."""

    def _parse_judge_assessment(self, response_content: str, judge_name: str) -> Dict:
        """Parse judge LLM's quality assessment response"""
        try:
            # Extract JSON from response
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_content)
            if json_match:
                assessment_data = json.loads(json_match.group(1).strip())
            else:
                # Try parsing the entire response as JSON
                assessment_data = json.loads(response_content.strip())

            # Add judge metadata
            assessment_data["judge_name"] = judge_name
            assessment_data["assessment_method"] = "LLM_Judge"

            self.logger.info(f"✅ Successfully parsed assessment from {judge_name}")
            return assessment_data

        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse JSON from {judge_name}: {str(e)}")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "judge_name": judge_name,
                "overall_quality_score": 0,
                "suitable_for_extraction": False,
                "raw_response": response_content[:500]  # First 500 chars for debugging
            }

    async def _validate_dimension_with_panel(self, validation_prompt: str,
                                           dimension: QualityValidationDimension) -> QualityValidationResult:
        """Use UQLM panel to validate a specific dimension"""

        try:
            dimension_start_time = time.time()
            self.logger.info(f"🕐 Starting UQLM panel validation for {dimension.value}")

            # Generate and score using UQLM panel
            results = await self.panel.generate_and_score(prompts=[validation_prompt])

            df_results = results.to_df()

            if df_results.empty:
                raise ValueError("No results from UQLM panel")

            result_row = df_results.iloc[0]
            response_content = result_row['response']

            # Extract confidence score
            confidence_score = self.extract_confidence_score(response_content)

            self.logger.info(f"📊 UQLM panel confidence score for {dimension.value}: {confidence_score}")

            # Try to extract structured JSON analysis
            try:
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_content)
                if json_match:
                    analysis_result = json.loads(json_match.group(1).strip())
                    self.logger.info(f"✅ Successfully extracted JSON from UQLM panel response for {dimension.value}")
                else:
                    # Try parsing the entire response as JSON
                    analysis_result = json.loads(response_content)
                    self.logger.info(f"✅ Successfully parsed JSON from UQLM panel response for {dimension.value}")

                # Extract structured results
                issues = analysis_result.get('issues', [])
                summary = analysis_result.get('summary', f'{dimension.value} analysis completed')
                reliability_level = analysis_result.get('reliability_level', 'medium')

            except json.JSONDecodeError:
                self.logger.warning(f"⚠️ UQLM panel response not in valid JSON format for {dimension.value}")

                # Extract issues from text manually
                issues = self._extract_issues_from_text(response_content)
                summary = f'UQLM panel analysis for {dimension.value} completed with {confidence_score:.2f} confidence score'
                reliability_level = 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low'

            dimension_time = time.time() - dimension_start_time
            self.logger.info(f"⏱️ UQLM panel validation for {dimension.value} completed in {dimension_time:.2f} seconds")

            return QualityValidationResult(
                dimension=dimension,
                confidence_score=confidence_score,
                issues=issues,
                summary=summary,
                raw_response=response_content,
                reliability_level=reliability_level
            )

        except Exception as e:
            self.logger.error(f"❌ Error in UQLM panel validation for {dimension.value}: {str(e)}")
            raise ValueError(f"UQLM panel validation failed for {dimension.value}: {str(e)}")

    def extract_confidence_score(self, response_content: str) -> float:
        """Extract confidence score from UQLM panel response"""
        try:
            # Look for explicit confidence score statement
            confidence_match = re.search(r"confidence_score:\s*([0-9]*\.?[0-9]+)", response_content, re.IGNORECASE)
            if confidence_match:
                return float(confidence_match.group(1))

            # Look for JSON confidence score
            json_match = re.search(r'"confidence_score":\s*([0-9]*\.?[0-9]+)', response_content)
            if json_match:
                return float(json_match.group(1))

            # Default fallback
            self.logger.warning("⚠️ Could not extract confidence score from response, using default 0.5")
            return 0.5

        except (ValueError, AttributeError) as e:
            self.logger.warning(f"⚠️ Error extracting confidence score: {str(e)}, using default 0.5")
            return 0.5

    def _extract_issues_from_text(self, text: str) -> List[str]:
        """Extract issues from unstructured text response"""
        issues = []

        # Look for bullet points or numbered lists
        issue_patterns = [
            r"[-•]\s*(.+)",
            r"\d+\.\s*(.+)",
            r"Issue:\s*(.+)",
            r"Problem:\s*(.+)"
        ]

        for pattern in issue_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            issues.extend([match.strip() for match in matches if match.strip()])

        # If no structured issues found, look for sentences with issue keywords
        if not issues:
            issue_keywords = ['issue', 'problem', 'error', 'incorrect', 'missing', 'inconsistent']
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in issue_keywords):
                    issues.append(sentence.strip())

        return issues[:5]  # Limit to top 5 issues

    def _create_error_result(self, dimension: QualityValidationDimension, error_msg: str) -> QualityValidationResult:
        """Create error result for dimension validation failure"""
        return QualityValidationResult(
            dimension=dimension,
            confidence_score=0.0,
            issues=[f"Validation error: {error_msg}"],
            summary=f"Validation failed for {dimension.value}",
            raw_response=f"Error: {error_msg}",
            reliability_level="low"
        )

    def _format_validation_results(self, validation_results: Dict, llm_assessment: Dict, image_path: str, judge_assessments: Dict = None) -> Dict[str, Any]:
        """Format comprehensive validation results"""

        # Calculate weighted confidence based on dimension importance
        weights = {
            QualityValidationDimension.VISUAL_ACCURACY.value: 0.30,
            QualityValidationDimension.QUANTITATIVE_RELIABILITY.value: 0.25,
            QualityValidationDimension.HALLUCINATION_DETECTION.value: 0.20,
            QualityValidationDimension.SEVERITY_ASSESSMENT.value: 0.15,
            QualityValidationDimension.CONSISTENCY_CHECK.value: 0.05,
            QualityValidationDimension.RECOMMENDATION_VALIDITY.value: 0.05
        }

        weighted_confidence = sum(
            result["confidence_score"] * weights.get(dim_key, 0.1)
            for dim_key, result in validation_results.items()
        ) / sum(weights.values())

        # Count high-reliability vs low-reliability dimensions
        low_reliability_count = sum(
            1 for result in validation_results.values()
            if result["reliability_level"] == "low"
        )

        # Collect all critical issues
        all_issues = []
        for result in validation_results.values():
            all_issues.extend(result["issues"])

        # Determine overall reliability
        if weighted_confidence >= 0.8 and low_reliability_count == 0:
            level = "HIGH"
            is_reliable = True
            recommendation = "LLM quality assessment is highly reliable and can be trusted for automated decisions."
        elif weighted_confidence >= 0.65 and low_reliability_count <= 1:
            level = "MEDIUM"
            is_reliable = True
            recommendation = "LLM quality assessment is generally reliable but review flagged issues before using."
        elif weighted_confidence >= 0.4 and low_reliability_count <= 2:
            level = "LOW"
            is_reliable = False
            recommendation = "LLM quality assessment has reliability concerns. Manual review required before use."
        else:
            level = "VERY_LOW"
            is_reliable = False
            recommendation = "LLM quality assessment has significant reliability issues. Consider using OpenCV assessment instead."

        # Calculate judge consensus if available
        judge_consensus = self._calculate_judge_consensus(judge_assessments) if judge_assessments else None

        result = {
            "validation_summary": {
                "overall_confidence": weighted_confidence,
                "is_reliable": is_reliable,
                "reliability_level": level,
                "critical_issues": all_issues,
                "recommendation": recommendation,
                "validated_dimensions_count": len(validation_results),
                "llm_overall_score": llm_assessment.get('overall_quality_score', 0),
                "llm_suitable_for_extraction": llm_assessment.get('suitable_for_extraction', False)
            },
            "dimensional_analysis": validation_results,
            "quality_metadata": {
                "panel_judges": len(self.judge_llms),
                "image_path": image_path,
                "validation_method": "UQLM LLMPanel",
                "llm_assessment_method": llm_assessment.get('assessment_method', 'LLM'),
                "llm_model_used": llm_assessment.get('model_used', 'Unknown')
            }
        }

        # Add judge assessments and consensus if available
        if judge_assessments:
            result["judge_assessments"] = judge_assessments
            if judge_consensus:
                result["judge_consensus"] = judge_consensus

        return result

    def _calculate_judge_consensus(self, judge_assessments: Dict) -> Dict[str, Any]:
        """Calculate consensus metrics from judge assessments"""
        if not judge_assessments:
            return None

        # Filter out error assessments
        valid_assessments = {k: v for k, v in judge_assessments.items() if "error" not in v}

        if not valid_assessments:
            return {"error": "No valid judge assessments available"}

        # Calculate consensus metrics
        scores = [assessment.get('overall_quality_score', 0) for assessment in valid_assessments.values()]
        suitable_votes = [assessment.get('suitable_for_extraction', False) for assessment in valid_assessments.values()]

        # Overall consensus
        avg_score = sum(scores) / len(scores) if scores else 0
        score_std = (sum((x - avg_score) ** 2 for x in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0
        suitable_consensus = sum(suitable_votes) / len(suitable_votes) if suitable_votes else 0

        # Issue-level consensus
        issue_consensus = {}
        quality_issues = ['blur_detection', 'contrast_assessment', 'glare_identification',
                         'water_stains', 'tears_or_folds', 'cut_off_detection',
                         'missing_sections', 'obstructions']

        for issue in quality_issues:
            issue_data = []
            for assessment in valid_assessments.values():
                if issue in assessment and isinstance(assessment[issue], dict):
                    issue_data.append(assessment[issue])

            if issue_data:
                detected_votes = [item.get('detected', False) for item in issue_data]
                severity_votes = [item.get('severity_level', 'low') for item in issue_data]
                confidence_scores = [item.get('confidence_score', 0) for item in issue_data]
                quantitative_measures = [item.get('quantitative_measure', 0) for item in issue_data]

                issue_consensus[issue] = {
                    "detection_consensus": sum(detected_votes) / len(detected_votes),
                    "avg_confidence": sum(confidence_scores) / len(confidence_scores),
                    "avg_quantitative_measure": sum(quantitative_measures) / len(quantitative_measures),
                    "severity_agreement": len(set(severity_votes)) == 1,  # True if all agree
                    "most_common_severity": max(set(severity_votes), key=severity_votes.count)
                }

        return {
            "judge_count": len(valid_assessments),
            "score_consensus": {
                "average_score": avg_score,
                "score_standard_deviation": score_std,
                "score_agreement": "high" if score_std < 1.0 else "medium" if score_std < 2.0 else "low"
            },
            "suitability_consensus": {
                "suitable_percentage": suitable_consensus * 100,
                "unanimous": suitable_consensus in [0.0, 1.0]
            },
            "issue_level_consensus": issue_consensus,
            "overall_agreement": "high" if score_std < 1.0 and (suitable_consensus in [0.0, 1.0]) else "medium" if score_std < 2.0 else "low"
        }

    def _create_error_validation_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result for complete validation failure"""
        return {
            "validation_summary": {
                "overall_confidence": 0.0,
                "is_reliable": False,
                "reliability_level": "ERROR",
                "critical_issues": [error_msg],
                "recommendation": "Fix system issues and retry validation",
                "validated_dimensions_count": 0,
                "llm_overall_score": 0,
                "llm_suitable_for_extraction": False
            },
            "dimensional_analysis": {},
            "quality_metadata": {
                "validation_method": "UQLM LLMPanel",
                "error": error_msg
            }
        }
