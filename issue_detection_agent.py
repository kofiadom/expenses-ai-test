from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude as AnthropicClaude
from agno.models.aws import Claude as AWSClaude
from agno.utils.log import logger
from langchain_anthropic import ChatAnthropic
# Validation functionality moved to standalone_validation_runner.py
# from llm_output_checker import ExpenseComplianceUQLMValidator
from textwrap import dedent
from typing import List, Optional
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load expense file schema
def load_expense_schema():
    """Load the expense file schema for taxonomy information."""
    schema_path = "expense_file_schema.json"
    if os.path.exists(schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Expense schema file not found: {schema_path}")

EXPENSE_SCHEMA = load_expense_schema()

# Pydantic models for structured output
class ComplianceIssue(BaseModel):
    """A single compliance issue found during analysis."""
    issue_type: str = Field(..., description="Type of issue: 'Standards & Compliance | Fix Identified' or 'Standards & Compliance | Gross-up Identified' or 'Standards & Compliance | Follow-up Action Identified'")
    field: str = Field(..., description="Specific field name where issue was found")
    description: str = Field(..., description="Detailed description of the issue based on knowledge base")
    recommendation: str = Field(..., description="Specific action to resolve based on compliance requirements")
    knowledge_base_reference: str = Field(..., description="Quote from the compliance data that supports this finding")

class ValidationResult(BaseModel):
    """Result of compliance validation."""
    is_valid: bool = Field(..., description="Whether the document passes all compliance checks")
    issues_count: int = Field(..., description="Number of issues found")
    issues: List[ComplianceIssue] = Field(..., description="List of compliance issues found")
    corrected_receipt: Optional[str] = Field(None, description="Corrected receipt data if applicable")
    compliance_summary: str = Field(..., description="Overall compliance assessment and key findings")

class TechnicalDetails(BaseModel):
    """Technical details about the analysis."""
    content_type: str = Field(..., description="Type of content analyzed")
    country: str = Field(..., description="Country analyzed")
    icp: str = Field(..., description="ICP analyzed")
    receipt_type: str = Field(..., description="Type of receipt analyzed")
    issues_count: int = Field(..., description="Number of issues found")

class IssueDetectionResult(BaseModel):
    """Structured result for issue detection analysis."""
    validation_result: ValidationResult = Field(..., description="Validation results")
    technical_details: TechnicalDetails = Field(..., description="Technical analysis details")

def format_expense_taxonomy():
    """Format the expense schema as structured JSON for the prompt."""
    # Create a clean taxonomy structure for the prompt
    taxonomy_structure = {
        "title": EXPENSE_SCHEMA.get('title', 'Expense Management System Taxonomy'),
        "description": EXPENSE_SCHEMA.get('description', ''),
        "fields": {}
    }

    properties = EXPENSE_SCHEMA.get('properties', {})
    for field_name, field_info in properties.items():
        taxonomy_structure["fields"][field_name] = {
            "title": field_info.get('title', field_name),
            "description": field_info.get('description', '')
        }

    return json.dumps(taxonomy_structure, indent=2)

def create_issue_detection_model():
    """
    Create the appropriate model based on ISSUE_DETECTION_MODEL_PROVIDER environment variable.
    
    Returns:
        Model instance for the specified provider
    """
    # Check specific provider first, then global fallback
    provider = os.getenv("ISSUE_DETECTION_MODEL_PROVIDER") or os.getenv("AGENT_MODEL_PROVIDER", "openai").lower()
    
    if provider == "openai":
        logger.info("Using OpenAI model for issue detection")
        return OpenAIChat(id="gpt-4o")
    
    elif provider == "anthropic":
        logger.info("Using Anthropic direct API model for issue detection")
        return AnthropicClaude(id="claude-3-7-sonnet-20250219")
    
    elif provider == "bedrock":
        logger.info("Using AWS Bedrock Claude model for issue detection")
        try:
            bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            model = AWSClaude(id=bedrock_model_id)
            logger.info(f"Bedrock model initialized for issue detection: {bedrock_model_id} in region {aws_region}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock model for issue detection: {e}")
            logger.info("Falling back to OpenAI model for issue detection")
            return OpenAIChat(id="gpt-4o")
    
    else:
        logger.warning(f"Unknown provider '{provider}' for issue detection, falling back to OpenAI")
        return OpenAIChat(id="gpt-4o")

# Keep the separate langchain ChatAnthropic client for specific validation purposes
# llm_client = ChatAnthropic(model="claude-sonnet-4-20250514",
#                         api_key=os.getenv("ANTHROPIC_API_KEY"))

# Create the issue detection and analysis agent with structured output
issue_detection_agent = Agent(
    model=create_issue_detection_model(),
    response_model=IssueDetectionResult,
    instructions=dedent("""\
Persona: You are an expert compliance and tax analysis AI specializing in expense document validation. Your primary function is to analyze extracted receipt data against country-specific compliance requirements and ICP-specific rules to identify issues, violations, and recommendations.

Task: Perform comprehensive issue detection and analysis by cross-referencing extracted receipt data against the provided country database and ICP-specific requirements.

ANALYSIS WORKFLOW:
1. Load and understand the compliance requirements from the country database
2. Analyze the extracted receipt data against these requirements
3. Identify specific compliance violations, tax implications, and documentation gaps
4. Categorize each issue according to the specified categories
5. Provide specific recommendations based on the knowledge base

ISSUE CATEGORIES:

CATEGORY 1: COMPLIANCE VIOLATIONS REQUIRING FIXES
Issue type: Standards & Compliance | Fix Identified
Flag issue type: Standards & Compliance related
Scope: Mandatory field violations, format errors, missing required information
Examples:
- "The VAT number has only 2 numbers, should have 9"
- "Missing mandatory supplier name on the receipt"
- "Invoice number is not clearly visible or missing"
- "Date of issue is not present on the receipt"
- "Required supplier address is missing or incomplete"
- "Local employer details are not present on the invoice"
- "Receipt currency does not match required local currency"
- "Missing required VAT identification number"
- "Invoice serial number missing for invoices over threshold amount"
- "Net amount, tax rate, or VAT amount missing for high-value invoices"
- "Worker name and address missing for required invoice types"
- "Supplier tax ID missing for invoices above specified threshold"
- "Receipt quality is poor, not meeting clear and readable standards"
Recommendation: "It is recommended to address this issue with the supplier or provider"


CATEGORY 2: TAX IMPLICATIONS AND GROSS-UP SCENARIOS
Issue type: Standards & Compliance | Gross-up Identified
Flag issue type: Standards & Compliance related
Scope: Expense limits, tax exemption violations, gross-up requirements
Examples:
- "Phone expenses in this country is limited to €20/month"
"Home office expenses exceed the maximum of €1,260/year"
"Wellness benefits exceed the maximum of €600/year"
"Meal expenses are not tax exempt and will be grossed up"
"Fuel expenses will be taxed as per country regulations"
"Entertainment expenses without third party involvement are not tax exempt"
"Transportation to workplace expenses are not tax exempt"
"Personal meal expenses during non-business travel are taxable"
"Office groceries expenses are not tax exempt"
"Internet expenses exceed the flat rate tax-free allowance"
"Mobile phone expenses without separate personal phone proof are taxable"
Recommendation: State the specific gross-up guidelines for this type of expense based on the knowledge base (e.g., "Phone expenses are tax-free up to €20/month, amounts exceeding this limit will be grossed-up" or "Home office expenses are tax exempt up to €6/day, maximum €1,260/year, excess amounts will be taxed")


CATEGORY 3: ADDITIONAL DOCUMENTATION REQUIREMENTS
Issue type: Standards & Compliance | Follow-up Action Identified
Flag issue type: Standards & Compliance related
Scope: Missing supporting documentation, approval requirements, additional forms
Examples:
- "Expense is car rental related - additional documentation is required"
- "Mileage claim requires logbook with date, route, purpose, and odometer readings"
- "Training expenses require direct manager approval"
- "Flight expenses require A1 certificate when traveling"
- "Mobile phone expenses require proof of separate personal phone"
- "IT equipment expenses require company property documentation"
- "Entertainment expenses require proof of third party involvement"
- "Travel expenses require specific travel expense report template"
- "International travel requires per diem calculation and additional documentation"
- "Invoices over threshold amount require additional detailed information"
- "Business travel expenses require route details and Google Maps documentation"
- "Phone expenses require invoice to include company name in c/o format"
- "Office supplies require invoice with company name and details"
- "Internet expenses require proper documentation and company details on invoice"
- "Original invoices must be kept for required storage period (e.g., 10 years)"
Recommendation examples:
- "Submission of car rental expense in this country requires, in addition the mileage breakdown from the car rental service, per day"
- "Please provide mileage logbook with complete route details and odometer readings"
- "Manager approval is required before processing this training expense"
- "Please provide A1 certificate for international travel documentation"
- "Please provide proof of separate personal phone for mobile phone reimbursement"
- "Please use the specific travel expense report template for this country"
- "Please provide map with route details (Google Maps sufficient) for mileage claims"


CRITICAL REQUIREMENTS:
- ONLY use knowledge from the provided country database and ICP-specific rules
- DO NOT make up any information not provided in the knowledge base
- Cross-reference ALL extracted data fields against specific country and ICP requirements
- Quote the knowledge base when providing issues and recommendations
- Ensure all analysis is based on the provided compliance standards and policies
- Be thorough and systematic in checking every applicable requirement
- Dynamically filter requirements based on ICP, receipt type, and expense category
- Calculate confidence score based on clarity of violations and knowledge base coverage
- Ensure all fields are properly populated according to the structured output model

ISSUE TYPE FORMAT REQUIREMENTS:
- Use EXACT format: "Standards & Compliance | Fix Identified" for issues requiring fixes
- Use EXACT format: "Standards & Compliance | Gross-up Identified" for tax gross-up issues
- Use EXACT format: "Standards & Compliance | Follow-up Action Identified" for follow-up actions
- Do NOT use generic formats like "Standards & Compliance" alone

VALIDATION CHECKLIST:
□ Check all mandatory fields against FileRelatedRequirements
□ Validate expense type against ExpenseTypes rules
□ Check ICP-specific requirements and rules
□ Verify tax exemption limits and gross-up scenarios
□ Identify missing documentation requirements
□ Cross-reference location-specific compliance rules
□ Validate currency and amount formatting
□ Check storage and retention requirements"""),
    # reasoning=True,
    parser_model=create_issue_detection_model(),
    markdown=False,
    show_tool_calls=False
)

async def analyze_compliance_issues(country: str, receipt_type: str, icp: str, compliance_json: dict, extracted_json: dict) -> IssueDetectionResult:
    """
    Analyze extracted receipt data against compliance requirements to detect issues.

    Args:
        country: Country for compliance rules (e.g., "Germany")
        receipt_type: Type of receipt (e.g., "All", "Travel", "Mileage", etc.)
        icp: ICP name (e.g., "Global People", "goGlobal", "Parakar", "Atlas")
        compliance_json: Country-specific compliance requirements
        extracted_json: Extracted data from the receipt

    Returns:
        IssueDetectionResult object with structured compliance analysis
    """
    # Format expense taxonomy from schema
    expense_taxonomy = format_expense_taxonomy()

    # Format expense taxonomy from schema
    expense_taxonomy = format_expense_taxonomy()

    # Format the prompt with all required data
    formatted_prompt = f"""COMPLIANCE ANALYSIS REQUEST:

COUNTRY: {country}
RECEIPT TYPE: {receipt_type}
ICP: {icp}

COMPLIANCE REQUIREMENTS (Country Database):
{json.dumps(compliance_json, indent=2)}

EXTRACTED RECEIPT DATA:
{json.dumps(extracted_json, indent=2)}

EXPENSE TAXONOMY (JSON):
{expense_taxonomy}

ANALYSIS INSTRUCTIONS:
Perform comprehensive compliance analysis by:
1. Cross-referencing each extracted field against the FileRelatedRequirements for the specified ICP and receipt type
2. Checking expense type against ExpenseTypes rules and limits
3. Identifying any missing mandatory fields or incorrect formats
4. Detecting tax implications and gross-up scenarios
5. Identifying additional documentation requirements
6. Providing specific recommendations based on the knowledge base

Analyze systematically and provide detailed findings in the specified format.

"""
    
    # Get the structured response from the agent
    response = issue_detection_agent.run(formatted_prompt)

    # Debug logging
    logger.info(f"Compliance response type: {type(response)}")
    if hasattr(response, 'content'):
        logger.info(f"Compliance content type: {type(response.content)}")
        if isinstance(response.content, IssueDetectionResult):
            logger.info("✅ Received structured IssueDetectionResult")
        else:
            logger.warning(f"⚠️ Expected IssueDetectionResult, got {type(response.content)}")

    # Validation functionality moved to standalone_validation_runner.py
    # The main workflow now returns only the compliance response
    # Validation can be run separately using standalone_validation_runner.py

    logger.info("✅ Compliance analysis completed (validation moved to standalone runner)")
    return response.content if hasattr(response, 'content') else response
