from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from textwrap import dedent
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Create the issue detection and analysis agent
issue_detection_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    #model=Claude(id="claude-3-7-sonnet-20250219"),
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
- VAT number format violations (incorrect digit count, format)
- Missing mandatory supplier information (name, address, VAT ID)
- Missing invoice/receipt identifiers or serial numbers
- Date format issues or missing transaction dates
- Currency mismatches with local requirements
- Missing tax information for high-value invoices
- Poor receipt quality affecting readability
- Missing worker details for specific invoice types
- Incomplete supplier tax identification
Recommendation: "It is recommended to address this issue with the supplier or provider"

CATEGORY 2: TAX IMPLICATIONS AND GROSS-UP SCENARIOS
Issue type: Standards & Compliance | Gross-up Identified
Flag issue type: Standards & Compliance related
Scope: Expense limits, tax exemption violations, gross-up requirements
Examples:
- Expenses exceeding tax-free limits (phone €20/month, home office €1,260/year, wellness €600/year)
- Non-tax-exempt expenses (personal meals, office groceries, entertainment without third party)
- Transportation to workplace expenses
- Internet expenses exceeding flat rate allowances
- Mobile phone expenses without personal phone proof
- Fuel and vehicle expenses subject to taxation
Recommendation: State specific gross-up guidelines from knowledge base with exact limits and tax implications

CATEGORY 3: ADDITIONAL DOCUMENTATION REQUIREMENTS
Issue type: Standards & Compliance | Follow-up Action Identified
Flag issue type: Standards & Compliance related
Scope: Missing supporting documentation, approval requirements, additional forms
Examples:
- Mileage claims requiring detailed logbooks
- Training expenses requiring manager approval
- Travel expenses requiring A1 certificates or travel templates
- Car rental requiring additional mileage documentation
- Entertainment requiring third party proof
- IT equipment requiring property documentation
- International travel requiring per diem calculations
- Storage period compliance for original documents
Recommendation: Specify exact documentation requirements and procedures from knowledge base

CRITICAL REQUIREMENTS:
- ONLY use knowledge from the provided country database and ICP-specific rules
- DO NOT make up any information not provided in the knowledge base
- Cross-reference ALL extracted data fields against specific country and ICP requirements
- Quote the knowledge base when providing issues and recommendations
- Ensure all analysis is based on the provided compliance standards and policies
- Be thorough and systematic in checking every applicable requirement
- Dynamically filter requirements based on ICP, receipt type, and expense category
- Calculate confidence score based on clarity of violations and knowledge base coverage
- Your output MUST BE ONLY a valid JSON object matching the specified structure

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
  "validation_result": {
    "is_valid": true/false,
    "issues_count": number,
    "issues": [
      {
        "issue_type": "Standards & Compliance | Fix Identified/Gross-up Identified/Follow-up Action Identified",
        "field": "specific_field_name",
        "description": "Detailed description of the issue based on knowledge base",
        "recommendation": "Specific action to resolve based on compliance requirements",
        "knowledge_base_reference": "Quote from the compliance data that supports this finding"
      }
    ],
    "corrected_receipt": null,
    "compliance_summary": "Overall compliance assessment and key findings"
  },
  "technical_details": {
    "content_type": "ReceiptValidationResult",
    "country": "analyzed_country",
    "icp": "analyzed_icp",
    "receipt_type": "analyzed_receipt_type",
    "issues_count": number_of_issues,
    "has_reasoning": true
  }
}

VALIDATION CHECKLIST:
□ Check all mandatory fields against FileRelatedRequirements
□ Validate expense type against ExpenseTypes rules
□ Check ICP-specific requirements and rules
□ Verify tax exemption limits and gross-up scenarios
□ Identify missing documentation requirements
□ Cross-reference location-specific compliance rules
□ Validate currency and amount formatting
□ Check storage and retention requirements"""),
    reasoning=True,
    markdown=False,
    show_tool_calls=False
)

def analyze_compliance_issues(country: str, receipt_type: str, icp: str, compliance_json: dict, extracted_json: dict) -> str:
    """
    Analyze extracted receipt data against compliance requirements to detect issues.
    
    Args:
        country: Country for compliance rules (e.g., "Germany")
        receipt_type: Type of receipt (e.g., "All", "Travel", "Mileage", etc.)
        icp: ICP name (e.g., "Global People", "goGlobal", "Parakar", "Atlas")
        compliance_json: Country-specific compliance requirements
        extracted_json: Extracted data from the receipt
        
    Returns:
        JSON string with detailed issue analysis
    """
    # Format the prompt with all required data
    formatted_prompt = f"""COMPLIANCE ANALYSIS REQUEST:

COUNTRY: {country}
RECEIPT TYPE: {receipt_type}
ICP: {icp}

COMPLIANCE REQUIREMENTS (Country Database):
{json.dumps(compliance_json, indent=2)}

EXTRACTED RECEIPT DATA:
{json.dumps(extracted_json, indent=2)}

ANALYSIS INSTRUCTIONS:
Perform comprehensive compliance analysis by:
1. Cross-referencing each extracted field against the FileRelatedRequirements for the specified ICP and receipt type
2. Checking expense type against ExpenseTypes rules and limits
3. Identifying any missing mandatory fields or incorrect formats
4. Detecting tax implications and gross-up scenarios
5. Identifying additional documentation requirements
6. Providing specific recommendations based on the knowledge base

Analyze systematically and provide detailed findings in the specified format."""
    
    # Get the response from the agent
    response = issue_detection_agent.run(formatted_prompt)

    # Debug logging
    print(f"DEBUG: Compliance response type: {type(response)}")
    print(f"DEBUG: Compliance response has content: {hasattr(response, 'content')}")
    if hasattr(response, 'content'):
        print(f"DEBUG: Compliance content type: {type(response.content)}")
        print(f"DEBUG: Compliance content length: {len(response.content) if response.content else 'None'}")
        print(f"DEBUG: Compliance content preview: {response.content[:200] if response.content else 'None'}")

    return response




