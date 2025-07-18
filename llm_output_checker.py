import os
import json
import re
import logging
import time
from langchain_anthropic import ChatAnthropic
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Import uqlm LLMPanel - based on the working example
try:
    from uqlm import LLMPanel
    UQLM_AVAILABLE = True
except ImportError:
    print("Warning: UQLM library not available. Install with: pip install uqlm")
    LLMPanel = None
    UQLM_AVAILABLE = False

class ValidationDimension(Enum):
    FACTUAL_GROUNDING = "factual_grounding"
    KNOWLEDGE_BASE_ADHERENCE = "knowledge_base_adherence" 
    COMPLIANCE_ACCURACY = "compliance_accuracy"
    ISSUE_CATEGORIZATION = "issue_categorization"
    RECOMMENDATION_VALIDITY = "recommendation_validity"
    HALLUCINATION_DETECTION = "hallucination_detection"

@dataclass
class ComplianceValidationResult:
    dimension: ValidationDimension
    confidence_score: float  # 0.0 to 1.0
    issues: List[str]
    summary: str
    raw_response: str
    reliability_level: str  # 'high', 'medium', 'low'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dimension": self.dimension.value if hasattr(self.dimension, 'value') else str(self.dimension),
            "confidence_score": self.confidence_score,
            "issues": self.issues,
            "summary": self.summary,
            "raw_response": self.raw_response,
            "reliability_level": self.reliability_level,
            "total_issues": len(self.issues)
        }


class ExpenseComplianceUQLMValidator:
    """
    UQLM-based compliance validator for expense analysis responses.
    Validates AI compliance analysis against source data using LLM-as-a-Judge approach.
    """

    def __init__(self, primary_llm, logger: logging.Logger = None):
        """
        Initialize UQLM compliance validator with judge panel.
        
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
            self.logger.info("🎯 UQLM LLM Panel initialized successfully with 3 judges for compliance validation")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize UQLM panel: {str(e)}")
            self.panel = None

    def extract_confidence_score(self, text: str) -> float:
        """
        Extracts the confidence score from a string using regex.
        Returns a float if found, otherwise raises an error.
        """
        # Try multiple patterns for confidence extraction
        patterns = [
            r'"?confidence[_ ]?score"?\s*[:=]\s*([0-9.]+)',
            r'confidence[:\s]+([0-9.]+)',
            r'I am ([0-9.]+)%?\s*confident',
            r'([0-9.]+)%?\s*confidence'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 range if needed
                return score / 100.0 if score > 1.0 else score
        
        raise ValueError("confidence_score not found in the string")

    async def validate_compliance_response(self, 
                                         ai_response: str,
                                         country: str,
                                         receipt_type: str,
                                         icp: str,
                                         compliance_json: Dict,
                                         extracted_json: Dict) -> Dict[str, Any]:
        """
        Validate AI's compliance analysis response using UQLM panel
        
        Args:
            ai_response: The AI's compliance analysis response (JSON string)
            country: Country for compliance rules
            receipt_type: Type of receipt
            icp: ICP name
            compliance_json: Country-specific compliance requirements
            extracted_json: Extracted receipt data
            
        Returns:
            Comprehensive validation results with UQLM uncertainty quantification
        """
        if not self.panel:
            raise ValueError("UQLM panel not properly initialized")

        validation_start_time = time.time()
        self.logger.info("🕐 Starting UQLM-based compliance validation")
        
        # Parse the AI response to understand its structure
        try:
            # Handle markdown-wrapped JSON
            content = ai_response.strip() if ai_response else ""
            if content.startswith('```json') and content.endswith('```'):
                content = content[7:-3].strip()  # Remove ```json and ```
            elif content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()  # Remove ``` and ```

            parsed_response = json.loads(content)
            self.logger.info("✅ Successfully parsed AI compliance response")
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse AI response as JSON: {str(e)}")
            self.logger.error(f"Raw AI response content: {ai_response[:500] if ai_response else 'None'}")
            return self._create_error_validation_result("Invalid JSON format in AI response")

        # Validate each dimension using UQLM panel
        validation_results = {}
        
        for dimension in ValidationDimension:
            self.logger.info(f"🔍 Validating {dimension.value} using UQLM panel")
            
            try:
                # Create dimension-specific validation prompt
                validation_prompt = self._create_validation_prompt(
                    ai_response, parsed_response, country, receipt_type, icp,
                    compliance_json, extracted_json, dimension
                )
                
                # Use UQLM panel for validation
                dimension_result = await self._validate_dimension_with_panel(
                    validation_prompt, dimension
                )
                validation_results[dimension.value] = dimension_result
                
            except Exception as e:
                self.logger.error(f"❌ Error validating {dimension.value}: {str(e)}")
                validation_results[dimension.value] = self._create_error_result(dimension, str(e))

        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(validation_results)
        
        formatted_res = {
            "validation_summary": {
                "overall_confidence": overall_assessment["confidence"],
                "is_reliable": overall_assessment["is_reliable"],
                "reliability_level": overall_assessment["level"],
                "critical_issues": overall_assessment["critical_issues"],
                "recommendation": overall_assessment["recommendation"],
                "validated_issues_count": len(parsed_response.get("validation_result", {}).get("issues", [])),
                "ai_confidence": parsed_response.get("validation_result", {}).get("confidence_score", 0.0)
            },
            "dimensional_analysis": validation_results,
            "compliance_metadata": {
                "panel_judges": len(self.judge_llms),
                "country": country,
                "receipt_type": receipt_type,
                "icp": icp,
                "validation_method": "UQLM LLMPanel",
                "original_issues_found": len(parsed_response.get("validation_result", {}).get("issues", [])),
                "ai_reported_confidence": parsed_response.get("validation_result", {}).get("confidence_score", 0.0)
            }
        }

        validation_time = time.time() - validation_start_time
        self.logger.info(f"⏱️ UQLM compliance validation completed in {validation_time:.2f} seconds")
        self.logger.info(f"Validation results: {formatted_res}")
        return formatted_res

    def _create_validation_prompt(self, ai_response: str, parsed_response: Dict, 
                                country: str, receipt_type: str, icp: str,
                                compliance_json: Dict, extracted_json: Dict, 
                                dimension: ValidationDimension) -> str:
        """Create validation prompt for a specific dimension"""
        
        dimension_instructions = {
            ValidationDimension.FACTUAL_GROUNDING: {
                "focus": "Factual Grounding Analysis",
                "description": "Verify that ALL facts, rules, and requirements cited in the AI response are actually present in the provided compliance database and extracted receipt data.",
                "checks": [
                    "Are all compliance rules cited actually found in the FileRelatedRequirements?",
                    "Are all VAT numbers, addresses, and company names correctly quoted from the database?",
                    "Are all extracted receipt fields accurately referenced?",
                    "Are there any made-up facts or requirements not in the source data?"
                ]
            },
            ValidationDimension.KNOWLEDGE_BASE_ADHERENCE: {
                "focus": "Knowledge Base Adherence",
                "description": "Ensure all recommendations and issue categorizations strictly follow the provided compliance database rules.",
                "checks": [
                    "Are all 'knowledge_base_reference' quotes actually found in the compliance data?",
                    "Do the issue types match the categories defined in the system prompt?",
                    "Are recommendations properly based on the compliance requirements?",
                    "Are ICP-specific rules correctly applied?"
                ]
            },
            ValidationDimension.COMPLIANCE_ACCURACY: {
                "focus": "Compliance Accuracy",
                "description": "Validate that identified compliance violations are correct and complete based on the provided data.",
                "checks": [
                    "Are all identified violations actually violations according to the compliance rules?",
                    "Are there any obvious violations that were missed?",
                    "Are the field names and descriptions accurate?",
                    "Is the compliance logic correctly applied?"
                ]
            },
            ValidationDimension.ISSUE_CATEGORIZATION: {
                "focus": "Issue Categorization",
                "description": "Verify that issues are correctly categorized according to the three defined categories.",
                "checks": [
                    "Are 'Fix Identified' issues properly categorized?",
                    "Are 'Gross-up Identified' issues correctly identified?",
                    "Are 'Follow-up Action Identified' issues appropriate?",
                    "Do the issue types match the actual problems found?"
                ]
            },
            ValidationDimension.RECOMMENDATION_VALIDITY: {
                "focus": "Recommendation Validity", 
                "description": "Assess whether recommendations are appropriate and actionable based on the compliance requirements.",
                "checks": [
                    "Are recommendations specific and actionable?",
                    "Do recommendations align with the knowledge base guidance?",
                    "Are the recommended actions appropriate for the identified issues?",
                    "Are any recommendations missing or inappropriate?"
                ]
            },
            ValidationDimension.HALLUCINATION_DETECTION: {
                "focus": "Hallucination Detection",
                "description": "Identify any fabricated information, made-up rules, or invented requirements.",
                "checks": [
                    "Are there any invented compliance rules or limits?",
                    "Are all numerical values and thresholds from the actual database?",
                    "Are there any fictional policy requirements?",
                    "Are company names, addresses, and VAT numbers correctly quoted?"
                ]
            }
        }
        
        dim_info = dimension_instructions[dimension]
        checks_text = "\n".join([f"- {check}" for check in dim_info["checks"]])
        
        return f"""
                COMPLIANCE VALIDATION TASK: {dim_info["focus"]}

                VALIDATION OBJECTIVE:
                {dim_info["description"]}

                SPECIFIC VALIDATION CHECKS:
                {checks_text}

                CONTEXT INFORMATION:
                Country: {country}
                Receipt Type: {receipt_type}  
                ICP: {icp}

                SOURCE DATA - COMPLIANCE REQUIREMENTS:
                {json.dumps(compliance_json, indent=2)}

                SOURCE DATA - EXTRACTED RECEIPT:
                {json.dumps(extracted_json, indent=2)}

                AI COMPLIANCE ANALYSIS TO VALIDATE:
                {ai_response}

                EXTRACTED ISSUES FROM AI RESPONSE:
                {json.dumps(parsed_response.get("validation_result", {}).get("issues", []), indent=2)}

                AI REPORTED CONFIDENCE: {parsed_response.get("validation_result", {}).get("confidence_score", "Not specified")}

                VALIDATION INSTRUCTIONS:
                1. Focus specifically on the {dimension.value.replace('_', ' ').title()} dimension
                2. Cross-reference EVERY fact, rule, and requirement against the source data
                3. Identify any discrepancies, hallucinations, or missing validations
                4. Provide a confidence score (0.0-1.0) for your assessment
                5. List specific issues if any are found

                SCORING GUIDELINES:
                - 0.9-1.0: Excellent adherence, no significant issues
                - 0.7-0.8: Good adherence, minor issues only  
                - 0.5-0.6: Moderate issues requiring attention
                - 0.3-0.4: Significant problems affecting reliability
                - 0.0-0.2: Critical failures, major hallucinations

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

    async def _validate_dimension_with_panel(self, validation_prompt: str, 
                                           dimension: ValidationDimension) -> ComplianceValidationResult:
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

            return ComplianceValidationResult(
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

    def _extract_issues_from_text(self, text: str) -> List[str]:
        """Extract issues from unstructured text response"""
        issues = []
        
        # Look for common issue patterns
        issue_patterns = [
            r"(?:issue|problem|violation|error|discrepancy):\s*(.+)",
            r"(?:found|detected|identified):\s*(.+)",
            r"(?:missing|incorrect|invalid|fabricated):\s*(.+)",
            r"(?:hallucination|made-up|invented):\s*(.+)"
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            issues.extend([match.strip() for match in matches])
        
        # Remove duplicates and limit
        unique_issues = list(dict.fromkeys(issues))
        return unique_issues[:5] # Limit to 5 unique issues max
        #return unique_issues[:10]  # Limit to 10 issues max

    def _create_error_result(self, dimension: ValidationDimension, error_msg: str) -> ComplianceValidationResult:
        """Create error result for failed validation"""
        return ComplianceValidationResult(
            dimension=dimension,
            confidence_score=0.0,
            issues=[f"Validation failed: {error_msg}"],
            summary=f"Error in {dimension.value} validation",
            raw_response=f"Error: {error_msg}",
            reliability_level="low"
        )

    def _create_error_validation_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result for complete validation failure"""
        return {
            "validation_summary": {
                "overall_confidence": 0.0,
                "is_reliable": False,
                "reliability_level": "ERROR",
                "critical_issues": [error_msg],
                "recommendation": "Fix system issues and retry validation",
                "validated_issues_count": 0,
                "ai_confidence": 0.0
            },
            "dimensional_analysis": {},
            "compliance_metadata": {
                "validation_method": "UQLM LLMPanel",
                "error": error_msg
            }
        }

    def _calculate_overall_assessment(self, validation_results: Dict[str, ComplianceValidationResult]) -> Dict[str, Any]:
        """Calculate overall assessment from dimensional results"""
        if not validation_results:
            return {
                "confidence": 0.0,
                "is_reliable": False,
                "level": "ERROR",
                "critical_issues": ["No validation results"],
                "recommendation": "Validation failed - check system configuration"
            }
        
        # Calculate weighted confidence based on dimension importance
        weights = {
            ValidationDimension.FACTUAL_GROUNDING.value: 0.25,
            ValidationDimension.KNOWLEDGE_BASE_ADHERENCE.value: 0.25,
            ValidationDimension.HALLUCINATION_DETECTION.value: 0.20,
            ValidationDimension.COMPLIANCE_ACCURACY.value: 0.15,
            ValidationDimension.ISSUE_CATEGORIZATION.value: 0.10,
            ValidationDimension.RECOMMENDATION_VALIDITY.value: 0.05
        }
        
        weighted_confidence = sum(
            result.confidence_score * weights.get(dim_key, 0.1)
            for dim_key, result in validation_results.items()
        ) / sum(weights.values())
        
        # Count high-reliability vs low-reliability dimensions
        low_reliability_count = sum(
            1 for result in validation_results.values()
            if result.reliability_level == "low"
        )
        
        # Collect critical issues
        critical_issues = []
        for result in validation_results.values():
            if result.reliability_level in ["low", "medium"]:
                critical_issues.extend(result.issues)
        
        # Determine overall reliability
        if weighted_confidence >= 0.8 and low_reliability_count == 0:
            level = "HIGH"
            is_reliable = True
            recommendation = "AI response is highly reliable and can be trusted for compliance decisions."
        elif weighted_confidence >= 0.65 and low_reliability_count <= 1:
            level = "MEDIUM"
            is_reliable = True
            recommendation = "AI response is generally reliable but review flagged issues before using."
        elif weighted_confidence >= 0.4 and low_reliability_count <= 2:
            level = "LOW"
            is_reliable = False
            recommendation = "AI response has reliability concerns. Manual review required before use."
        else:
            level = "VERY_LOW"
            is_reliable = False
            recommendation = "AI response has significant reliability issues. Consider regenerating or manual analysis."
        
        return {
            "confidence": weighted_confidence,
            "is_reliable": is_reliable,
            "level": level,
            "critical_issues": critical_issues[:10],  # Limit to most critical
            "recommendation": recommendation
        }