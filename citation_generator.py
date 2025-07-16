"""
Citation Generator

This module generates citations for extracted data by finding where field names 
and values appear in the source documents using LLM analysis.
"""

import json
import os
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from textwrap import dedent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create the citation agent
citation_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions=dedent("""\
You are a citation expert specializing in finding where extracted data fields and their values appear in source documents.

Your task is to analyze structured output from data extraction and find TWO types of citations for each field:

1. FIELD CITATION: Where does this field name/concept appear in the source?
   - Check extraction requirements for field_type definitions
   - Check markdown for field labels, headers, form fields
   - Look for: "Total:", "Supplier Name:", table headers, section labels, etc.

2. VALUE CITATION: Where does this exact value appear in the source?
   - Find exact matches in markdown text
   - Handle fuzzy matches for dates, numbers, currencies
   - Consider context and formatting variations
   - Look for values near field labels or in structured sections

ANALYSIS APPROACH:
- Use semantic understanding to match field concepts even with different wording
- Handle variations in formatting (dates, currencies, numbers)
- Assess confidence based on match quality and context
- Provide surrounding context for validation

OUTPUT FORMAT:
Return a valid JSON object with this exact structure:

{
  "citations": {
    "field_name": {
      "field_citation": {
        "source_text": "text found in source",
        "confidence": 0.9,
        "source_location": "requirements|markdown", 
        "context": "surrounding text for validation",
        "match_type": "exact|fuzzy|contextual"
      },
      "value_citation": {
        "source_text": "value found in source",
        "confidence": 0.8,
        "source_location": "markdown",
        "context": "surrounding text for validation", 
        "match_type": "exact|fuzzy|contextual"
      }
    }
  },
  "metadata": {
    "total_fields_analyzed": 0,
    "fields_with_field_citations": 0,
    "fields_with_value_citations": 0,
    "average_confidence": 0.0
  }
}

CRITICAL: Your response must be ONLY valid JSON. No explanatory text, markdown formatting, or additional content.
"""),
    markdown=False,
    show_tool_calls=False
)


def generate_citations(structured_output: dict, extraction_requirements: str, markdown_content: str, filename: str) -> dict:
    """
    Generate citations using LLM analysis of structured output vs source documents.
    
    Args:
        structured_output: JSON result from extract_data_from_receipt()
        extraction_requirements: Compliance JSON string used for extraction  
        markdown_content: Markdown text used for extraction
        filename: For saving citation file
        
    Returns:
        Citation analysis results
    """
    try:
        logger.info(f"Generating citations for {filename}")
        
        # Prepare the prompt with all three inputs
        prompt = f"""STRUCTURED OUTPUT (JSON):
{json.dumps(structured_output, indent=2)}

EXTRACTION REQUIREMENTS (JSON):
{extraction_requirements}

MARKDOWN TEXT:
{markdown_content}

Analyze the structured output and find field and value citations in the source documents."""

        # Get citation analysis from LLM
        response = citation_agent.run(prompt)
        
        # Handle different response formats
        if hasattr(response, 'content'):
            content = response.content
            if content is None or content.strip() == "":
                raise ValueError("Empty content returned from citation agent")

            # Handle markdown-wrapped JSON
            content = content.strip()
            if content.startswith('```json') and content.endswith('```'):
                content = content[7:-3].strip()
            elif content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()

            response_text = content
        else:
            response_text = str(response)
        
        # Parse the citation results
        try:
            citations = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from citation agent: {e}")
            logger.error(f"Raw response: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON response from citation agent: {e}")
        
        # Save citations to file
        save_citations(citations, filename)
        
        logger.info(f"Citations generated successfully for {filename}")
        return citations
        
    except Exception as e:
        logger.error(f"Citation generation error for {filename}: {e}")
        return {
            "citations": {},
            "metadata": {
                "error": str(e),
                "total_fields_analyzed": 0,
                "fields_with_field_citations": 0,
                "fields_with_value_citations": 0,
                "average_confidence": 0.0
            }
        }


def save_citations(citations: dict, filename: str):
    """
    Save citation results to file.
    
    Args:
        citations: Citation analysis results
        filename: Base filename (without extension)
    """
    try:
        # Create citation folder if it doesn't exist
        citation_folder = Path("citation_folder")
        citation_folder.mkdir(exist_ok=True)
        
        # Save citation file
        citation_file = citation_folder / f"{filename}_citation.json"
        with open(citation_file, 'w', encoding='utf-8') as f:
            json.dump(citations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Citations saved to {citation_file}")
        
    except Exception as e:
        logger.error(f"Failed to save citations for {filename}: {e}")


def get_citation_stats(citations: dict) -> dict:
    """
    Get statistics about citation quality.
    
    Args:
        citations: Citation analysis results
        
    Returns:
        Citation statistics
    """
    if not citations or "citations" not in citations:
        return {"error": "No citation data available"}
    
    citation_data = citations["citations"]
    total_fields = len(citation_data)
    
    field_citations = 0
    value_citations = 0
    total_confidence = 0.0
    confidence_count = 0
    
    for field_name, field_citations_data in citation_data.items():
        if "field_citation" in field_citations_data:
            field_citations += 1
            if "confidence" in field_citations_data["field_citation"]:
                total_confidence += field_citations_data["field_citation"]["confidence"]
                confidence_count += 1
                
        if "value_citation" in field_citations_data:
            value_citations += 1
            if "confidence" in field_citations_data["value_citation"]:
                total_confidence += field_citations_data["value_citation"]["confidence"]
                confidence_count += 1
    
    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
    
    return {
        "total_fields": total_fields,
        "fields_with_field_citations": field_citations,
        "fields_with_value_citations": value_citations,
        "field_citation_rate": field_citations / total_fields if total_fields > 0 else 0.0,
        "value_citation_rate": value_citations / total_fields if total_fields > 0 else 0.0,
        "average_confidence": round(avg_confidence, 2)
    }
