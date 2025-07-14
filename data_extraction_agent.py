from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from textwrap import dedent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Create the data extraction agent
data_extraction_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    #model=Claude(id="claude-3-7-sonnet-20250219"),
    instructions=dedent("""\
Persona: You are a meticulous and highly accurate data extraction AI. You specialize in parsing unstructured text from expense documents and structuring it into a precise JSON format. Your primary function is to identify and extract data points from the receipt text, not to validate them against specific rules.

Task: Your goal is to extract specific fields from the provided RECEIPT TEXT. You must use the EXTRACTION REQUIREMENTS as the definitive guide for what field types to look for. Additionally, identify and extract any other standard invoice or receipt fields not explicitly listed, including line items with their individual details.

INPUTS:
1. EXTRACTION REQUIREMENTS (JSON):
This JSON object defines the field types you must attempt to extract. Look at the "FieldType" values to understand what kinds of information to extract. The "Description" provides context to help you locate the correct information. IGNORE the "Rule" field completely - it is for validation purposes only and should not affect your extraction.

{extraction_requirements_json}

2. RECEIPT TEXT (MARKDOWN):
This is the raw text from the document that needs to be analyzed.

{receipt_text}

INSTRUCTIONS:
Analyze: Carefully read the RECEIPT TEXT to identify all available information.
Extract Field Types: For each unique "FieldType" in the EXTRACTION REQUIREMENTS, extract the actual value found in the receipt text. IGNORE any "Rule" specifications - extract what is actually present in the receipt.
Extract Additional Fields: Also extract any other relevant information found in the receipt that could be useful for expense management, such as:
- Line items with their details (products/services, quantities, prices)
- Transaction identifiers, reference numbers, or invoice numbers
- Date and time information
- Contact information (phone, email, website, etc.)
- Tax-related information (rates, amounts, tax IDs)
- Payment-related information
- Location or table identifiers
- Any special notes, terms, or conditions
- Subtotals, discounts, tips, or other financial breakdowns
- Any other structured data present in the receipt

Format: Structure your findings into a single, valid JSON object.
The keys of your output JSON MUST be the snake_case version of the "FieldType" values. For example, "Supplier Name" becomes "supplier_name", "VAT Number" becomes "vat_number".
For additional fields not in the extraction requirements, use descriptive snake_case field names that clearly indicate what the data represents.
If a field type cannot be found in the text, its value in the output JSON MUST be null. Do not guess or invent data.
For dates, standardize the format to YYYY-MM-DD. If you cannot determine the year, assume the current year.
For amounts and rates, extract only the numerical value (e.g., 120.50, 19.0).
For currency, use the standard 3-letter ISO code (e.g., "EUR", "USD") if possible; otherwise, extract the symbol.
For line items, create an array of objects with details like item name, quantity, unit price, and total price.
Adapt field names to the type of receipt (restaurant, hotel, transport, retail, etc.) while maintaining consistency.

CRITICAL REQUIREMENT:
Your final output MUST BE ONLY a valid JSON object. Do not include any explanatory text, greetings, apologies, or markdown formatting like ```json before or after the JSON object.
EXAMPLE OUTPUT STRUCTURE:
Include all fields from the extraction requirements (using snake_case of FieldType values) plus any additional relevant fields found in the receipt. Use descriptive field names for additional fields.

{
  "country": "Germany",
  "supplier_name": "THE SUSHI CLUB",
  "supplier_address": "Mohrenstr.42, 10117 Berlin",
  "vat_number": null,
  "currency": "EUR",
  "total_amount": 64.40,
  "date_of_issue": "2019-02-05",
  "line_items": [
    {
      "description": "Miso Soup",
      "quantity": 1,
      "unit_price": 3.90,
      "total_price": 3.90
    }
  ],
  "contact_phone": "+49 30 23 916 036",
  "contact_email": "info@thesushiclub.de",
  "contact_website": "WWW.TheSushiClub.de",
  "transaction_time": "23:10:54",
  "receipt_type": "Rechnung",
  "table_number": "24",
  "transaction_reference": "L0001 FRÜH",
  "special_notes": "TIP IS NOT INCLUDED",
  "tax_rate": null,
  "vat": null,
  "name": null,
  "address": null,
  "supplier": null,
  "expense": null,
  "route": null,
  "car_details": null,
  "purpose": null,
  "odometer_reading": null,
  "travel_date": null,
  "a1_certificate": null,
  "payment_receipt": null,
  "manager_approval": null,
  "personal_phone_proof": null,
  "storage_period": null
}"""),
    #reasoning=True,
    # Ensure clean JSON output
    markdown=False,
    show_tool_calls=False
)

# Example usage function
def extract_data_from_receipt(extraction_requirement_json: str, receipt_text: str) -> str:
    """
    Extract data from receipt text using the specified extraction field types and additional relevant fields.

    Args:
        extraction_requirement_json: JSON string defining the field types to extract
        receipt_text: The raw receipt text to analyze

    Returns:
        JSON string with extracted data including specified fields and line items
    """
    # Format the prompt with the actual data
    formatted_prompt = f"""EXTRACTION REQUIREMENTS (JSON):
{extraction_requirement_json}

RECEIPT TEXT (MARKDOWN):
{receipt_text}"""
    
    # Get the response from the agent
    response = data_extraction_agent.run(formatted_prompt)
    return response


