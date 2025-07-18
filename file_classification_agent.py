from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from textwrap import dedent
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Load expense file schema
def load_expense_schema():
    """Load the expense file schema for field-based classification."""
    schema_path = "expense_file_schema.json"
    if os.path.exists(schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Expense schema file not found: {schema_path}")

EXPENSE_SCHEMA = load_expense_schema()



# Create the file classification agent
file_classification_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    #model=Claude(id="claude-3-7-sonnet-20250219"),
    instructions=dedent("""\
Persona: You are an expert file classification AI specializing in expense document analysis. Your primary function is to determine if a file contains expense-related content and classify it appropriately.

Task: Analyze the provided text to determine:
1. Whether this is an expense document (Y/N)
2. If it's an expense, classify the expense type
3. Identify the document language and confidence level
4. Verify location consistency

CLASSIFICATION CRITERIA:

STEP 1: EXPENSE IDENTIFICATION (SCHEMA-BASED)
First determine: Is this file an expense? (Y/N)

Use the provided EXPENSE FILE SCHEMA to identify expense documents based on field presence.

Look for each schema field in the document. If you find 5 or more fields, it's an expense document.

REQUIRED FOR EXPENSE CLASSIFICATION:
- Evidence of payment completed (not just booking/reservation)
- Actual amounts charged/paid
- Payment confirmation or receipt of transaction

NOT EXPENSES (even if business-related):
- Booking confirmations without payment proof
- Reservation details without charges shown
- Quotes, estimates, or pending invoices
- Payment details on next page (incomplete documents)


EXPENSE TYPE CLUSTERS (classify only if is_expense = true):
- flights: airline tickets, boarding passes, flight bookings, airport services
- meals: restaurants, food delivery, catering, dining, coffee shops, bars
- accommodation: hotels, lodging, room bookings, Airbnb, hostels, resorts
- telecommunications: phone bills, internet services, mobile plans, data charges
- travel: transportation (taxi, rideshare, bus, train), car rental, fuel, parking, tolls
- training: courses, workshops, educational services, conferences, seminars, certifications
- mileage: vehicle expenses, fuel receipts, car maintenance, parking fees
- entertainment: events, shows, client entertainment, team activities, sports events
- office_supplies: stationery, equipment, software licenses, office furniture
- utilities: electricity, water, gas, heating, cooling services
- professional_services: consulting, legal, accounting, marketing, IT services
- medical: healthcare services, medical consultations, pharmacy purchases
- other: miscellaneous business expenses not fitting above categories

LANGUAGE IDENTIFICATION:
Identify the primary language of the document and provide a confidence score (0-100%).
Consider factors like:
- Vocabulary and word patterns
- Grammar structures
- Currency symbols and formats
- Address formats
- Common phrases and expressions
Minimum confidence threshold: 80%

LOCATION VERIFICATION:
Extract the country/location from the document (from addresses, phone codes, currency, etc.)
Compare with the expected location provided in the input.

ERROR CATEGORIES AND HANDLING:
1. "File cannot be processed"
   - When: Technical issues, corrupted text, unreadable content, empty files
   - Action: Set is_expense=false, error_type="File cannot be processed"

2. "File identified not as an expense"
   - When: Text identified but doesn't fit expense definitions per location
   - Action: Set is_expense=false, error_type="File identified not as an expense"

3. "File cannot be analysed"
   - When: Language confidence below 80% threshold
   - Action: Set is_expense=false, error_type="File cannot be analysed"

4. "File location is not same as project's location"
   - When: Document location ≠ expected location input
   - Action: Set error_type="File location is not same as project's location"
   - Note: This can still be an expense, just flag the location mismatch

PROCESSING WORKFLOW:
1. First check if content is readable and processable
2. Identify language and calculate confidence score
3. Determine if content represents an expense document
4. If expense, classify the expense type cluster
5. Extract document location information
6. Compare document location with expected location
7. Set appropriate error flags if any issues found

OUTPUT FORMAT:
Return a JSON object with the following structure:
{
  "is_expense": true/false,
  "expense_type": "category_name" or null,
  "language": "language_name",
  "language_confidence": 0-100,
  "document_location": "detected_country/location" or null,
  "expected_location": "provided_expected_location",
  "location_match": true/false,
  "error_type": null or "error_category",
  "error_message": null or "detailed_error_description",
  "classification_confidence": 0-100,
  "reasoning": "brief explanation of classification decision",
  "schema_field_analysis": {
    "fields_found": ["list of schema fields identified in document"],
    "fields_missing": ["list of schema fields not found in document"],
    "total_fields_found": number,
    "expense_identification_reasoning": "detailed explanation citing exact fields found/missing for expense determination"
  }
}

CRITICAL REQUIREMENTS:
- Your output MUST BE ONLY a valid JSON object
- Do not include explanatory text, greetings, or markdown formatting
- Be conservative in classification - when in doubt, mark as not an expense
- Follow the exact error categories specified
- Provide clear reasoning for your decision"""),
    #reasoning=True,
    markdown=False,
    show_tool_calls=False
)

def classify_file(receipt_text: str, expected_country: str = None) -> str:
    """
    Classify a file to determine if it's an expense document and categorize it using schema-based field analysis.

    Args:
        receipt_text: The raw text content to analyze
        expected_country: The expected country/location for validation (optional)

    Returns:
        JSON string with classification results including schema field analysis
    """
    # Create schema field descriptions for the prompt
    schema_fields_description = ""
    for field_name, field_info in EXPENSE_SCHEMA.get("properties", {}).items():
        title = field_info.get("title", field_name)
        description = field_info.get("description", "")
        schema_fields_description += f"\n**{field_name}** ({title}):\n{description}\n"

    # Format the prompt with the actual data and schema
    formatted_prompt = f"""EXPENSE FILE SCHEMA FIELDS:
{schema_fields_description}

DOCUMENT TEXT TO ANALYZE:
{receipt_text}

EXPECTED LOCATION: {expected_country if expected_country else "Not specified"}

ANALYSIS INSTRUCTIONS:
1. Carefully examine the document text for each of the 8 schema fields listed above
2. For each field, determine if it is PRESENT or ABSENT in the document
3. Use the field descriptions and recognition patterns to guide your analysis
4. Count the total number of fields found
5. Apply the expense identification logic (3-4+ fields = expense)
6. Provide detailed reasoning citing the exact fields found/missing

Analyze the above text following the schema-based workflow and provide classification results in the specified JSON format."""

    # Get the response from the agent
    response = file_classification_agent.run(formatted_prompt)

    # Debug logging
    print(f"DEBUG: Classification response type: {type(response)}")
    print(f"DEBUG: Classification response has content: {hasattr(response, 'content')}")
    if hasattr(response, 'content'):
        print(f"DEBUG: Classification content type: {type(response.content)}")
        print(f"DEBUG: Classification content length: {len(response.content) if response.content else 'None'}")
        print(f"DEBUG: Classification content preview: {response.content[:200] if response.content else 'None'}")

    return response

