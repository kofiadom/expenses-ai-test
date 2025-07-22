"""
Citation Generator

This module generates citations for extracted data by finding where field names 
and values appear in the source documents using LLM analysis.
Supports multiple model providers: OpenAI, Anthropic, and AWS Bedrock.
"""

import json
import os
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude as AnthropicClaude
from agno.models.aws import Claude as AWSClaude
from agno.utils.log import logger
from textwrap import dedent
from dotenv import load_dotenv
from typing import Dict, Optional
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Pydantic models for structured output
class CitationInfo(BaseModel):
    """Information about a single citation."""
    source_text: str = Field(..., description="Text found in source")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    source_location: str = Field(..., description="Source location: requirements|markdown")
    context: str = Field(..., description="Surrounding text for validation")
    match_type: str = Field(..., description="Match type: exact|fuzzy|contextual")

class FieldCitation(BaseModel):
    """Citation information for a single field."""
    field_citation: Optional[CitationInfo] = Field(None, description="Citation for field name/concept")
    value_citation: Optional[CitationInfo] = Field(None, description="Citation for field value")

class CitationMetadata(BaseModel):
    """Metadata about citation analysis."""
    total_fields_analyzed: int = Field(..., description="Total number of fields analyzed")
    fields_with_field_citations: int = Field(..., description="Number of fields with field citations")
    fields_with_value_citations: int = Field(..., description="Number of fields with value citations")
    average_confidence: float = Field(..., description="Average confidence score")

class CitationResult(BaseModel):
    """Structured result for citation analysis."""
    citations: Dict[str, FieldCitation] = Field(..., description="Citations for each field")
    metadata: CitationMetadata = Field(..., description="Analysis metadata")

def create_citation_model():
    """
    Create the appropriate model based on CITATION_MODEL_PROVIDER environment variable.
    
    Returns:
        Model instance for the specified provider
    """
    provider = os.getenv("CITATION_MODEL_PROVIDER", "openai").lower()
    
    if provider == "openai":
        logger.info("Using OpenAI model for citations")
        return OpenAIChat(id="gpt-4o")
    
    elif provider == "anthropic":
        logger.info("Using Anthropic direct API model for citations")
        return AnthropicClaude(id="claude-3-5-sonnet-20241022")
    
    elif provider == "bedrock":
        logger.info("Using AWS Bedrock Claude model for citations")
        try:
            bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            model = AWSClaude(id=bedrock_model_id)
            logger.info(f"Bedrock model initialized: {bedrock_model_id} in region {aws_region}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock model: {e}")
            logger.info("Falling back to OpenAI model")
            return OpenAIChat(id="gpt-4o")
    
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to OpenAI")
        return OpenAIChat(id="gpt-4o")

# Create the citation agent with structured output
citation_agent = Agent(
    model=create_citation_model(),
    response_model=CitationResult,
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

CRITICAL REQUIREMENTS:
- Provide accurate citations with proper confidence scores
- Use semantic understanding to match field concepts
- Handle formatting variations appropriately
- Ensure all fields are properly populated according to the structured output model
"""),
    parser_model=create_citation_model(),
    markdown=False,
    show_tool_calls=False
)


def generate_citations(structured_output: dict, extraction_requirements: str, markdown_content: str, filename: str) -> CitationResult:
    """
    Generate citations using LLM analysis of structured output vs source documents.

    Args:
        structured_output: JSON result from extract_data_from_receipt()
        extraction_requirements: Compliance JSON string used for extraction
        markdown_content: Markdown text used for extraction
        filename: For saving citation file

    Returns:
        CitationResult object with structured citation analysis
    """
    try:
        provider = os.getenv("CITATION_MODEL_PROVIDER", "openai").lower()
        logger.info(f"Generating citations for {filename} using {provider} model")
        
        # Prepare the prompt with all three inputs
        prompt = f"""STRUCTURED OUTPUT (JSON):
{json.dumps(structured_output, indent=2)}

EXTRACTION REQUIREMENTS (JSON):
{extraction_requirements}

MARKDOWN TEXT:
{markdown_content}

Analyze the structured output and find field and value citations in the source documents."""

        # Get structured citation analysis from LLM
        response = citation_agent.run(prompt)

        # Handle structured response
        if hasattr(response, 'content'):
            citations = response.content
            logger.debug(f"Citation response content type: {type(citations)}")
            logger.debug(f"Citation response content: {citations}")

            if citations is None:
                logger.warning(f"⚠️ Citation response content is None")
                citations = CitationResult(
                    citations={},
                    metadata=CitationMetadata(
                        total_fields_analyzed=0,
                        fields_with_field_citations=0,
                        fields_with_value_citations=0,
                        average_confidence=0.0
                    )
                )
            elif not isinstance(citations, CitationResult):
                logger.warning(f"⚠️ Expected CitationResult, got {type(citations)}")
                # Fallback to creating a basic structure
                citations = CitationResult(
                    citations={},
                    metadata=CitationMetadata(
                        total_fields_analyzed=0,
                        fields_with_field_citations=0,
                        fields_with_value_citations=0,
                        average_confidence=0.0
                    )
                )
        else:
            logger.error("No content in citation response")
            citations = CitationResult(
                citations={},
                metadata=CitationMetadata(
                    total_fields_analyzed=0,
                    fields_with_field_citations=0,
                    fields_with_value_citations=0,
                    average_confidence=0.0
                )
            )

        # Ensure citations is never None
        if citations is None:
            logger.error(f"Citations is None after processing, creating fallback")
            citations = CitationResult(
                citations={},
                metadata=CitationMetadata(
                    total_fields_analyzed=0,
                    fields_with_field_citations=0,
                    fields_with_value_citations=0,
                    average_confidence=0.0
                )
            )

        # Save citations to file (convert to dict for JSON serialization)
        save_citations(citations.model_dump(), filename)

        logger.info(f"✅ Citations generated successfully for {filename} using {provider}")
        return citations
        
    except Exception as e:
        logger.error(f"Citation generation error for {filename}: {e}")
        return CitationResult(
            citations={},
            metadata=CitationMetadata(
                total_fields_analyzed=0,
                fields_with_field_citations=0,
                fields_with_value_citations=0,
                average_confidence=0.0
            )
        )


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


def get_citation_stats(citations) -> dict:
    """
    Get statistics about citation quality.

    Args:
        citations: Citation analysis results (dict or CitationResult)

    Returns:
        Citation statistics
    """
    # Handle both structured CitationResult and legacy dict format
    if hasattr(citations, 'model_dump'):
        # Convert CitationResult to dict
        citations_dict = citations.model_dump()
    elif isinstance(citations, dict):
        citations_dict = citations
    else:
        return {"error": "Invalid citation data format"}

    if not citations_dict or "citations" not in citations_dict:
        return {"error": "No citation data available"}

    citation_data = citations_dict["citations"]
    if citation_data is None:
        return {"error": "Citation data is None"}
    total_fields = len(citation_data)
    
    field_citations = 0
    value_citations = 0
    total_confidence = 0.0
    confidence_count = 0
    
    for _, field_citations_data in citation_data.items():
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
