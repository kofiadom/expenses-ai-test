#!/usr/bin/env python3
"""
Streamlit Demo Interface for Expense Processing System
"""

import streamlit as st
import asyncio
import json
import os
import pathlib
import tempfile
import time
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF for PDF preview
from agno.utils.log import logger
from expense_processing_workflow import ExpenseProcessingWorkflow
from image_quality_processor import ImageQualityProcessor
from llm_image_quality_assessor import LLMImageQualityAssessor
from standalone_validation_runner import StandaloneValidationRunner

def make_json_serializable(obj, _seen=None, _depth=0):
    """Convert objects to JSON-serializable format with recursion protection."""
    if _seen is None:
        _seen = set()

    # Prevent infinite recursion with depth limit
    if _depth > 10:
        return f"<Max depth reached for {type(obj).__name__}>"

    # Check for circular references
    obj_id = id(obj)
    if obj_id in _seen:
        return f"<Circular reference to {type(obj).__name__}>"

    try:
        # Try direct JSON serialization first
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # Add to seen set for recursion protection
    _seen.add(obj_id)

    try:
        if hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            result = {}
            for key, value in obj.__dict__.items():
                try:
                    result[key] = make_json_serializable(value, _seen, _depth + 1)
                except Exception:
                    result[key] = str(value)  # Fallback to string representation
            return result
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                try:
                    result[key] = make_json_serializable(value, _seen, _depth + 1)
                except Exception:
                    result[key] = str(value)  # Fallback to string representation
            return result
        elif isinstance(obj, (list, tuple)):
            result = []
            for item in obj:
                try:
                    result.append(make_json_serializable(item, _seen, _depth + 1))
                except Exception:
                    result.append(str(item))  # Fallback to string representation
            return result
        elif hasattr(obj, 'value'):
            # Handle enum values
            return obj.value
        else:
            # Fallback to string representation for unknown types
            return str(obj)
    finally:
        # Remove from seen set when done
        _seen.discard(obj_id)

# Page configuration
st.set_page_config(
    page_title="Expense Processing System Demo",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_available_countries():
    """Get list of available countries from data directory."""
    data_dir = pathlib.Path("data")
    countries = []

    if data_dir.exists():
        for json_file in data_dir.glob("*.json"):
            country_name = json_file.stem.title()  # Convert filename to title case
            countries.append(country_name)

    return sorted(countries) if countries else ["Germany"]  # Fallback

def initialize_session_state():
    """Initialize session state variables."""
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def analyze_uploaded_file_quality(uploaded_file):
    """Analyze the quality of an uploaded image file using both OpenCV and LLM assessment systems."""
    try:
        # Save uploaded file temporarily for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize quality processors
        opencv_processor = ImageQualityProcessor(document_type='receipt')
        llm_assessor = LLMImageQualityAssessor()

        # Perform OpenCV quality assessment
        opencv_results = opencv_processor.assess_image_quality(tmp_file_path)

        # Perform LLM quality assessment
        try:
            llm_assessment = llm_assessor.assess_image_quality_sync(tmp_file_path)
            llm_results = llm_assessor.format_assessment_for_workflow(llm_assessment, tmp_file_path)
        except Exception as llm_error:
            logger.warning(f"LLM quality assessment failed for {uploaded_file.name}: {str(llm_error)}")
            llm_results = {
                "error": f"LLM assessment failed: {str(llm_error)}",
                "assessment_method": "LLM",
                "overall_quality_score": 0,
                "suitable_for_extraction": False
            }

        # Return both results
        combined_results = {
            "opencv_assessment": opencv_results,
            "llm_assessment": llm_results,
            # Keep original format for backward compatibility
            "quality_score": opencv_results.get("quality_score", 0),
            "quality_level": opencv_results.get("quality_level", "Unknown"),
            "quality_passed": opencv_results.get("quality_passed", False),
            "overall_assessment": opencv_results.get("overall_assessment", {})
        }

        # Clean up temporary file
        pathlib.Path(tmp_file_path).unlink()

        return combined_results

    except Exception as e:
        logger.error(f"Image quality analysis failed for {uploaded_file.name}: {str(e)}")
        return {
            "error": str(e),
            "quality_score": 0.0,
            "quality_level": "Error",
            "quality_passed": False,
            "overall_assessment": {
                "score": 0.0,
                "level": "Error",
                "pass_fail": False,
                "issues_summary": [f"Analysis failed: {str(e)}"],
                "recommendations": ["Please try uploading the image again"]
            }
        }

def save_quality_results_to_files(quality_results_dict, save_directory="quality_reports"):
    """
    Save quality assessment results to individual JSON files in the same directory as main.py

    Args:
        quality_results_dict: Dictionary of quality results keyed by file identifiers
        save_directory: Directory to save the files (default: "quality_reports" - same as main.py)

    Returns:
        List of saved file paths
    """
    try:
        # Create save directory (same as main.py uses)
        save_path = pathlib.Path(save_directory)
        save_path.mkdir(exist_ok=True)

        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for file_key, quality_result in quality_results_dict.items():
            # Extract filename from file_key (format: "filename_size")
            filename = file_key.rsplit('_', 1)[0]  # Remove size suffix
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()

            # Create output filename (consistent with main.py format)
            output_filename = f"{safe_filename}_quality_{timestamp}.json"
            output_path = save_path / output_filename

            # Save quality results with JSON serialization fix
            with open(output_path, 'w') as f:
                serializable_results = json.loads(json.dumps(quality_result, default=str))
                json.dump(serializable_results, f, indent=2)

            saved_files.append(output_path)
            logger.info(f"💾 Saved quality results: {output_path}")

        return saved_files

    except Exception as e:
        logger.error(f"❌ Failed to save quality results: {str(e)}")
        return []


async def run_standalone_validation():
    """Run standalone LLM-as-judge validation on existing results."""
    try:
        # Initialize validation runner
        runner = StandaloneValidationRunner(
            results_dir="results",
            quality_dir="llm_quality_reports",
            validation_output_dir="streamlit_validation_results"
        )

        # Run validation
        summary = await runner.run_validation(
            validate_compliance=True,
            validate_quality=True
        )

        return summary, None

    except Exception as e:
        logger.error(f"Standalone validation failed: {str(e)}")
        return None, str(e)


def display_file_preview(uploaded_file):
    """Display preview of uploaded file (image or PDF)."""
    file_type = uploaded_file.type

    try:
        if file_type.startswith('image/'):
            # Display image preview
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Preview: {uploaded_file.name}", width=400)

        elif file_type == 'application/pdf':
            # Display PDF preview (first page)
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc[0]  # Get first page
                pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8))  # Reduced scale for smaller preview
                img_data = pix.tobytes("png")

                st.image(img_data, caption=f"Preview: {uploaded_file.name} (Page 1)", width=400)

                if len(doc) > 1:
                    st.info(f"📄 PDF has {len(doc)} pages. Showing page 1 only.")
            doc.close()

        else:
            # For other file types, show file info
            st.info(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")
            st.write("Preview not available for this file type.")

    except Exception as e:
        st.warning(f"Could not preview {uploaded_file.name}: {str(e)}")
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")

async def process_uploaded_files(uploaded_files, country, icp, llamaparse_api_key):
    """Process uploaded files through the expense workflow."""

    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        dataset_path = temp_path / "dataset"
        dataset_path.mkdir(exist_ok=True)

        # Save existing LLM quality results to avoid re-running assessments
        if hasattr(st.session_state, 'image_quality_results') and st.session_state.image_quality_results:
            llm_quality_dir = temp_path / "llm_quality_reports"
            llm_quality_dir.mkdir(exist_ok=True)

            for uploaded_file in uploaded_files:
                if uploaded_file.type.startswith('image/'):
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    quality_result = st.session_state.image_quality_results.get(file_key, {})

                    # Extract LLM assessment if available
                    llm_assessment = quality_result.get('llm_assessment')
                    if llm_assessment and 'error' not in llm_assessment:
                        # Save LLM quality result in the format expected by workflow
                        base_name = pathlib.Path(uploaded_file.name).stem
                        llm_quality_file = llm_quality_dir / f"llm_quality_{base_name}.json"

                        with open(llm_quality_file, 'w', encoding='utf-8') as f:
                            json.dump(llm_assessment, f, indent=2, ensure_ascii=False)

                        logger.info(f"💾 Reusing LLM quality assessment for {uploaded_file.name}")

        # Save uploaded files to temporary directory
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_path / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            logger.info(f"Saved uploaded file: {uploaded_file.name}")

            # Create dataset metadata for each file
            base_name = uploaded_file.name.rsplit('.', 1)[0] if '.' in uploaded_file.name else uploaded_file.name
            dataset_entry = {
                "filepath": uploaded_file.name,
                "country": country,
                "icp": icp,
                "receipt_type": "unknown",  # Will be determined by classification
                "description": f"Uploaded file: {uploaded_file.name}"
            }

            # Save dataset entry
            dataset_file = dataset_path / f"{base_name}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_entry, f, indent=2, ensure_ascii=False)

        # Create workflow
        workflow = ExpenseProcessingWorkflow(
            session_id=f"streamlit-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            debug_mode=False
        )

        # Process files
        progress_placeholder = st.empty()

        try:
            async for response in workflow.process_expenses(
                dataset_dir=str(dataset_path),
                llamaparse_api_key=llamaparse_api_key,
                input_folder=str(temp_path)
            ):
                progress_placeholder.info(f"🔄 {response.content}")

            # Get results from session state
            processing_results = workflow.session_state.get("processing_results", [])

            return processing_results, None

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return [], str(e)

def display_classification_result(classification):
    """Display classification results in a formatted way."""
    if not classification:
        st.warning("No classification data available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Is Expense", "✅ Yes" if classification.get('is_expense') else "❌ No")
        expense_type = classification.get('expense_type') or 'N/A'
        st.metric("Expense Type", expense_type.title() if expense_type != 'N/A' else 'N/A')
    
    with col2:
        st.metric("Language", classification.get('language', 'N/A'))
        st.metric("Language Confidence", f"{classification.get('language_confidence', 0)}%")
    
    with col3:
        st.metric("Location Match", "✅ Yes" if classification.get('location_match') else "❌ No")
        st.metric("Classification Confidence", f"{classification.get('classification_confidence', 0)}%")
    
    if classification.get('reasoning'):
        st.text_area("Classification Reasoning", classification['reasoning'], height=100)

    # Display schema field analysis if available
    schema_analysis = classification.get('schema_field_analysis')
    if schema_analysis:
        st.subheader("🔍 Schema Field Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            fields_found = schema_analysis.get('fields_found', [])
            st.metric("Fields Found", f"{len(fields_found)}/8")
            if fields_found:
                st.write("**✅ Found:**")
                for field in fields_found:
                    st.write(f"• {field}")

        with col2:
            fields_missing = schema_analysis.get('fields_missing', [])
            if fields_missing:
                st.write("**❌ Missing:**")
                for field in fields_missing[:5]:  # Show max 5 to save space
                    st.write(f"• {field}")
                if len(fields_missing) > 5:
                    st.write(f"• ... and {len(fields_missing) - 5} more")

        with col3:
            total_found = schema_analysis.get('total_fields_found', 0)
            threshold_met = total_found >= 5  # Based on updated threshold
            st.metric("Threshold Met", "✅ Yes" if threshold_met else "❌ No")
            st.caption(f"Requires 5+ fields for expense")

        # Schema-based reasoning
        schema_reasoning = schema_analysis.get('expense_identification_reasoning')
        if schema_reasoning:
            st.text_area("Schema-Based Reasoning", schema_reasoning, height=80)

def display_extraction_result(extraction):
    """Display extraction results in a comprehensive formatted way."""
    if not extraction:
        st.warning("No extraction data available")
        return

    # Main information in tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Supplier", "💰 Transaction", "📋 Line Items", "📄 Additional"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {extraction.get('supplier_name', 'N/A')}")
            st.write(f"**Address:** {extraction.get('supplier_address', 'N/A')}")
            st.write(f"**VAT Number:** {extraction.get('vat_number', 'N/A')}")
            st.write(f"**Tax Code:** {extraction.get('tax_code', 'N/A')}")
        with col2:
            st.write(f"**Company Reg:** {extraction.get('company_registration', 'N/A')}")
            st.write(f"**Cashier:** {extraction.get('cashier', 'N/A')}")
            st.write(f"**Location:** {extraction.get('location', 'N/A')}")
            st.write(f"**Register:** {extraction.get('cash_register', 'N/A')}")

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            total_amount = extraction.get('total_amount') or extraction.get('amount')
            st.write(f"**Total:** {extraction.get('currency', '')} {total_amount or 'N/A'}")
            st.write(f"**Currency:** {extraction.get('currency', 'N/A')}")
            st.write(f"**VAT:** {extraction.get('vat', 'N/A')}")
        with col2:
            st.write(f"**Date:** {extraction.get('date_of_issue', 'N/A')}")
            st.write(f"**Time:** {extraction.get('transaction_time', 'N/A')}")
            st.write(f"**Travel Date:** {extraction.get('travel_date', 'N/A')}")
        with col3:
            st.write(f"**Receipt Type:** {extraction.get('receipt_type', 'N/A')}")
            st.write(f"**Payment Method:** {extraction.get('payment_method', 'N/A')}")
            st.write(f"**Order Type:** {extraction.get('order_type', 'N/A')}")

    with tab3:
        # Line items
        if extraction.get('line_items'):
            line_items_data = []
            for item in extraction['line_items']:
                # Convert quantity to string to avoid Arrow serialization issues
                quantity = item.get('quantity', 'N/A')
                if quantity != 'N/A' and quantity is not None:
                    quantity = str(quantity)
                else:
                    quantity = 'N/A'

                line_items_data.append({
                    "Description": str(item.get('description', 'N/A')),
                    "Quantity": quantity,
                    "Unit Price": f"{extraction.get('currency', '')} {item.get('unit_price', 'N/A')}",
                    "Total": f"{extraction.get('currency', '')} {item.get('total_price', 'N/A')}",
                    "Date": str(item.get('date', 'N/A')),
                    "Item #": str(item.get('item_number', 'N/A')),
                    "Reference": str(item.get('reference', 'N/A'))
                })
            st.dataframe(line_items_data, use_container_width=True, height=300)
        else:
            st.info("No line items found")

    with tab4:
        # Additional fields in organized sections
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Info")
            st.write(f"**Name:** {extraction.get('name', 'N/A')}")
            st.write(f"**Personal Info:** {extraction.get('personal_information', 'N/A')}")
            st.write(f"**Order Code:** {extraction.get('order_code', 'N/A')}")
            st.write(f"**Bon Number:** {extraction.get('bon_number', 'N/A')}")

            st.subheader("Business Travel")
            st.write(f"**Purpose:** {extraction.get('purpose', 'N/A')}")
            st.write(f"**Route:** {extraction.get('route', 'N/A')}")
            st.write(f"**Manager Approval:** {extraction.get('manager_approval', 'N/A')}")
            st.write(f"**A1 Certificate:** {extraction.get('a1_certificate', 'N/A')}")

        with col2:
            st.subheader("Vehicle Info")
            st.write(f"**Car Details:** {extraction.get('car_details', 'N/A')}")
            st.write(f"**Odometer:** {extraction.get('odometer_reading', 'N/A')}")
            st.write(f"**Combined Mileage:** {extraction.get('combined_mileage', 'N/A')}")

            st.subheader("Other")
            st.write(f"**Tax Rate:** {extraction.get('tax_rate', 'N/A')}")
            st.write(f"**Storage Period:** {extraction.get('storage_period', 'N/A')}")
            st.write(f"**Phone Proof:** {extraction.get('personal_phone_proof', 'N/A')}")
            st.write(f"**Payment Receipt:** {extraction.get('payment_receipt', 'N/A')}")

        # Billing address if available
        billing_address = extraction.get('billing_address')
        if billing_address and isinstance(billing_address, dict):
            st.subheader("Billing Address")
            st.write(f"**Name:** {billing_address.get('name', 'N/A')}")
            st.write(f"**Address:** {billing_address.get('address', 'N/A')}")
            st.write(f"**City:** {billing_address.get('city', 'N/A')}, {billing_address.get('postal_code', 'N/A')}")
            st.write(f"**Country:** {billing_address.get('country', 'N/A')}")

def display_citation_result(citations):
    """Display citation results in a formatted way."""
    if not citations:
        st.info("No citation data available")
        return

    if "error" in citations.get("metadata", {}):
        st.error(f"Citation generation failed: {citations['metadata']['error']}")
        return

    # Display citation metadata
    metadata = citations.get("metadata", {})
    if metadata:
        st.subheader("📊 Citation Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Fields", metadata.get("total_fields_analyzed", 0))

        with col2:
            st.metric("Field Citations", metadata.get("fields_with_field_citations", 0))

        with col3:
            st.metric("Value Citations", metadata.get("fields_with_value_citations", 0))

        with col4:
            avg_confidence = metadata.get("average_confidence", 0)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Display individual citations
    citation_data = citations.get("citations", {})
    if citation_data:
        st.subheader("🔍 Field Citations")

        for field_name, field_citations in citation_data.items():
            with st.expander(f"📝 {field_name.replace('_', ' ').title()}", expanded=False):

                # Field citation
                field_citation = field_citations.get("field_citation", {})
                if field_citation:
                    st.write("**🏷️ Field Citation:**")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Found:** `{field_citation.get('source_text', 'N/A')}`")
                        st.write(f"**Context:** {field_citation.get('context', 'N/A')}")
                        st.write(f"**Location:** {field_citation.get('source_location', 'N/A')}")

                    with col2:
                        confidence = field_citation.get('confidence', 0)
                        match_type = field_citation.get('match_type', 'unknown')
                        st.metric("Confidence", f"{confidence:.2f}")
                        st.write(f"**Type:** {match_type}")

                # Value citation
                value_citation = field_citations.get("value_citation", {})
                if value_citation:
                    st.write("**💰 Value Citation:**")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Found:** `{value_citation.get('source_text', 'N/A')}`")
                        st.write(f"**Context:** {value_citation.get('context', 'N/A')}")
                        st.write(f"**Location:** {value_citation.get('source_location', 'N/A')}")

                    with col2:
                        confidence = value_citation.get('confidence', 0)
                        match_type = value_citation.get('match_type', 'unknown')
                        st.metric("Confidence", f"{confidence:.2f}")
                        st.write(f"**Type:** {match_type}")

                if not field_citation and not value_citation:
                    st.info("No citations found for this field")
    else:
        st.info("No field citations available")


def display_compliance_result(compliance):
    """Display compliance results in a formatted way."""
    if not compliance or 'validation_result' not in compliance:
        st.warning("No compliance data available")
        return

    validation = compliance['validation_result']

    col1, col2 = st.columns(2)

    with col1:
        is_valid = validation.get('is_valid', False)
        st.metric("Compliance Status", "✅ Valid" if is_valid else "❌ Issues Found")

    with col2:
        st.metric("Issues Count", validation.get('issues_count', 0))

    # Display issues
    if validation.get('issues'):
        st.subheader("Compliance Issues")
        for i, issue in enumerate(validation['issues'], 1):
            with st.expander(f"Issue {i}: {issue.get('issue_type', 'Unknown')}"):
                st.write(f"**Field:** {issue.get('field', 'N/A')}")
                st.write(f"**Description:** {issue.get('description', 'N/A')}")

                # Highlight the recommendation
                recommendation = issue.get('recommendation', 'N/A')
                if recommendation and recommendation != 'N/A':
                    st.info(f"**💡 Recommendation:** {recommendation}")
                else:
                    st.write(f"**Recommendation:** {recommendation}")

    # Display overall compliance summary and recommendation if available
    if compliance.get('compliance_summary'):
        st.subheader("📋 Compliance Summary")
        st.text_area("Summary", compliance['compliance_summary'], height=100, disabled=True)

    # Display overall recommendation if available at the top level
    if compliance.get('recommendation'):
        st.subheader("🎯 Overall Recommendation")
        st.info(compliance['recommendation'])

def load_validation_result(result_file_path):
    """Load UQLM validation results from separate validation file."""
    try:
        # Get the base name of the result file
        base_name = pathlib.Path(result_file_path).stem
        validation_file = pathlib.Path("validation_results") / f"{base_name}_validation.json"

        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading validation results: {e}")
        return None

def display_image_quality_result(image_quality, uploaded_file=None):
    """Display comprehensive image quality analysis results with image at top and assessments side by side."""
    if not image_quality:
        st.info("No image quality analysis available")
        return

    if "error" in image_quality:
        st.error(f"Image quality analysis failed: {image_quality['error']}")
        return

    # Display image at the top if available
    if uploaded_file and uploaded_file.type.startswith('image/'):
        st.write("**📷 Image Preview:**")
        try:
            image = Image.open(uploaded_file)
            # Resize image for display (max width 600px)
            max_width = 600
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height))
            st.image(image, caption=uploaded_file.name, use_column_width=False)
        except Exception as e:
            st.error(f"Could not display image: {str(e)}")

    # Check if we have both assessments
    has_opencv = "opencv_assessment" in image_quality
    has_llm = "llm_assessment" in image_quality

    if has_opencv and has_llm:
        # Display both assessments side by side
        st.write("**📸 Quality Analysis:**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔧 OpenCV Assessment**")
            opencv_data = image_quality["opencv_assessment"]
            display_opencv_quality_section(opencv_data)

        with col2:
            st.write("**🤖 LLM Assessment**")
            llm_data = image_quality["llm_assessment"]
            display_llm_quality_section(llm_data)

    else:
        # Fallback to original display for backward compatibility
        st.write("**📸 Quality Analysis:**")
        if has_opencv:
            opencv_data = image_quality["opencv_assessment"]
            display_opencv_quality_section(opencv_data)
        elif has_llm:
            llm_data = image_quality["llm_assessment"]
            display_llm_quality_section(llm_data)
        else:
            display_opencv_quality_section(image_quality)


def display_opencv_quality_section(opencv_data):
    """Display OpenCV quality assessment in your original format."""
    if "error" in opencv_data:
        st.error(f"OpenCV assessment failed: {opencv_data['error']}")
        return

    # Overall assessment (your original format)
    overall = opencv_data.get('overall_assessment', {})

    # Main metrics with smaller text (your original format)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = overall.get('score', 0)
        st.metric("Score", f"{score:.1f}/100")

    with col2:
        level = overall.get('level', 'Unknown')
        color = "🟢" if level in ['Excellent', 'Good'] else "🟡" if level == 'Acceptable' else "🔴"
        st.metric("Level", f"{color} {level}")

    with col3:
        passed = overall.get('pass_fail', False)
        st.metric("Status", "✅ PASS" if passed else "❌ FAIL")

    with col4:
        processing_time = opencv_data.get('processing_time_seconds', 0)
        st.metric("Time", f"{processing_time:.1f}s")

    # Score breakdown (your original format)
    score_breakdown = opencv_data.get('score_breakdown', {})
    if score_breakdown:
        st.subheader("📊 Score Breakdown")

        # Create a visual score breakdown with smaller text
        metrics_data = []
        for metric, data in score_breakdown.items():
            metrics_data.append({
                'Metric': metric.title(),
                'Score': f"{data.get('score', 0):.1f}",
                'Weight': f"{data.get('weight', 0)*100:.0f}%",
                'Contribution': f"{data.get('contribution', 0):.1f}"
            })

        st.dataframe(metrics_data, use_container_width=True, height=200)

    # Detailed results in tabs (your original format)
    detailed = opencv_data.get('detailed_results', {})
    if detailed:
        st.subheader("🔍 Details")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 Resolution", "🎯 Blur", "💡 Glare", "📏 Complete", "🩹 Damage"])

        with tab1:
            resolution = detailed.get('resolution', {})
            if resolution:
                col1, col2, col3 = st.columns(3)
                with col1:
                    dimensions = resolution.get('dimensions', {})
                    st.metric("Size", f"{dimensions.get('width', 0)}x{dimensions.get('height', 0)}")
                    st.metric("MP", f"{dimensions.get('megapixels', 0):.1f}")
                with col2:
                    dpi = resolution.get('dpi', {})
                    st.metric("DPI", f"{dpi.get('average', 0):.0f}")
                    quality = resolution.get('quality', {})
                    st.metric("Level", quality.get('level', 'Unknown'))
                with col3:
                    st.metric("Score", f"{resolution.get('quality', {}).get('score', 0):.1f}/100")
                    st.metric("Suitable", "✅ Yes" if resolution.get('quality', {}).get('suitable_for_ocr') else "❌ No")

        with tab2:
            blur = detailed.get('blur', {})
            if blur:
                col1, col2, col3 = st.columns(3)
                with col1:
                    metrics = blur.get('metrics', {})
                    st.metric("Score", f"{metrics.get('blur_score', 0):.1f}/100")
                    st.metric("Level", metrics.get('blur_level', 'Unknown'))
                with col2:
                    st.metric("Blurry", "❌ Yes" if metrics.get('is_blurry') else "✅ No")
                    st.metric("Variance", f"{metrics.get('laplacian_variance', 0):.1f}")
                with col3:
                    blur_types = blur.get('blur_types', {})
                    motion_blur = blur_types.get('motion_blur', {})
                    st.metric("Motion", "✅ Yes" if motion_blur.get('detected') else "❌ No")
                    if motion_blur.get('detected'):
                        st.metric("Direction", motion_blur.get('direction', 'Unknown'))

        with tab3:
            glare = detailed.get('glare', {})
            if glare:
                col1, col2, col3 = st.columns(3)
                with col1:
                    glare_analysis = glare.get('glare_analysis', {})
                    st.metric("Score", f"{glare_analysis.get('glare_score', 0):.1f}/100")
                    st.metric("Level", glare_analysis.get('glare_level', 'Unknown'))
                with col2:
                    st.metric("Spots", glare_analysis.get('num_glare_spots', 0))
                    st.metric("Coverage", f"{glare_analysis.get('glare_coverage_percent', 0):.1f}%")
                with col3:
                    brightness = glare.get('brightness_analysis', {})
                    st.metric("Brightness", f"{brightness.get('mean_brightness', 0):.1f}")
                    st.metric("Contrast", f"{brightness.get('contrast', 0):.1f}")

        with tab4:
            completeness = detailed.get('completeness', {})
            if completeness:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{completeness.get('completeness_score', 0):.1f}/100")
                    st.metric("Level", completeness.get('completeness_level', 'Unknown'))
                with col2:
                    corner_analysis = completeness.get('corner_analysis', {})
                    st.metric("Corners", f"{corner_analysis.get('visible_corners', 0)}/4")
                    st.metric("Shape", "✅ Rect" if corner_analysis.get('is_rectangular') else "❌ No")
                with col3:
                    boundary = completeness.get('boundary_analysis', {})
                    st.metric("Detected", "✅ Yes" if boundary.get('document_detected') else "❌ No")
                    st.metric("Cropping", boundary.get('cropping_quality', 'Unknown'))

        with tab5:
            damage = detailed.get('damage', {})
            if damage:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{damage.get('damage_score', 0):.1f}/100")
                    st.metric("Level", damage.get('damage_level', 'Unknown'))
                with col2:
                    damage_types = damage.get('damage_types', [])
                    st.metric("Types", len(damage_types))
                    if damage_types:
                        st.caption(", ".join(damage_types))
                with col3:
                    stain_analysis = damage.get('stain_analysis', {})
                    st.metric("Stains", f"{stain_analysis.get('stain_coverage', 0):.1f}%")
                    tear_analysis = damage.get('tear_analysis', {})
                    st.metric("Tears", f"{tear_analysis.get('tear_coverage', 0):.1f}%")

    # Issues and recommendations with smaller text
    col1, col2 = st.columns(2)

    with col1:
        if overall.get('issues_summary'):
            st.subheader("⚠️ Issues")
            for issue in overall['issues_summary']:
                st.warning(f"• {issue}")

    with col2:
        if overall.get('recommendations'):
            st.subheader("💡 Tips")
            for rec in overall['recommendations']:
                st.info(f"• {rec}")

    # Image type detection info (your original format)
    image_type = opencv_data.get('image_type_detection', {})
    if image_type:
        with st.expander("📱 Image Type Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Is Screenshot", "📱 Yes" if image_type.get('is_digital_screenshot') else "📷 No")
                st.metric("Image Subtype", image_type.get('image_subtype', 'Unknown'))
            with col2:
                st.metric("Detection Confidence", f"{image_type.get('confidence', 0)*100:.0f}%")


def display_llm_quality_section(llm_data):
    """Display LLM quality assessment results."""
    if "error" in llm_data:
        st.error(f"LLM assessment failed: {llm_data['error']}")
        return

    # Overall LLM assessment
    col1, col2, col3 = st.columns(3)

    with col1:
        score = llm_data.get('overall_quality_score', 0)
        st.metric("Score", f"{score}/10")

    with col2:
        suitable = llm_data.get('suitable_for_extraction', False)
        st.metric("Suitable for Extraction", "✅ Yes" if suitable else "❌ No")

    with col3:
        method = llm_data.get('assessment_method', 'LLM')
        model = llm_data.get('model_used', 'Unknown')
        st.metric("Method", f"🤖 {method}")
        st.caption(f"Model: {model}")

    # Quality issues breakdown
    st.subheader("🔍 Quality Issues Analysis")

    quality_issues = [
        ('blur_detection', '🌫️ Blur Detection'),
        ('contrast_assessment', '🌗 Contrast Assessment'),
        ('glare_identification', '💡 Glare Identification'),
        ('water_stains', '💧 Water Stains'),
        ('tears_or_folds', '📄 Tears/Folds'),
        ('cut_off_detection', '✂️ Cut-off Detection'),
        ('missing_sections', '🔍 Missing Sections'),
        ('obstructions', '🚫 Obstructions')
    ]

    # Create a table of quality issues
    issues_data = []
    for issue_key, issue_name in quality_issues:
        if issue_key in llm_data:
            issue_data = llm_data[issue_key]
            if isinstance(issue_data, dict):
                issues_data.append({
                    'Issue': issue_name,
                    'Detected': '✅ Yes' if issue_data.get('detected', False) else '❌ No',
                    'Severity': issue_data.get('severity_level', 'N/A').title(),
                    'Confidence': f"{issue_data.get('confidence_score', 0):.2f}",
                    'Measure': f"{issue_data.get('quantitative_measure', 0):.2f}",
                    'Description': issue_data.get('description', 'N/A')[:50] + '...' if len(issue_data.get('description', '')) > 50 else issue_data.get('description', 'N/A')
                })

    if issues_data:
        st.dataframe(issues_data, use_container_width=True, height=300)

        # Show detailed recommendations for detected issues
        detected_issues = [item for item in issues_data if item['Detected'] == '✅ Yes']
        if detected_issues:
            st.subheader("💡 LLM Recommendations")
            for issue_key, issue_name in quality_issues:
                if issue_key in llm_data:
                    issue_data = llm_data[issue_key]
                    if isinstance(issue_data, dict) and issue_data.get('detected', False):
                        recommendation = issue_data.get('recommendation', 'No recommendation available')
                        st.write(f"**{issue_name}:** {recommendation}")

    # Display UQLM validation results if available
    if 'uqlm_validation' in llm_data:
        st.write("---")  # Separator
        st.subheader("🎯 UQLM Validation Results")
        display_llm_quality_validation(llm_data['uqlm_validation'])


def display_llm_quality_validation(validation_data):
    """Display UQLM validation results for LLM quality assessment."""
    if not validation_data:
        st.info("No UQLM validation data available")
        return

    # Overall validation summary
    summary = validation_data.get('validation_summary', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence = summary.get('overall_confidence', 0)
        st.metric("Validation Confidence", f"{confidence:.2f}")

    with col2:
        reliability = summary.get('reliability_level', 'Unknown')
        color = "🟢" if reliability == "HIGH" else "🟡" if reliability == "MEDIUM" else "🔴"
        st.metric("Reliability", f"{color} {reliability}")

    with col3:
        is_reliable = summary.get('is_reliable', False)
        st.metric("Is Reliable", "✅ Yes" if is_reliable else "❌ No")

    with col4:
        validated_dims = summary.get('validated_dimensions_count', 0)
        st.metric("Dimensions Validated", validated_dims)

    # Recommendation
    recommendation = summary.get('recommendation', 'No recommendation available')
    if recommendation:
        st.info(f"**💡 Recommendation:** {recommendation}")

    # Critical issues
    critical_issues = summary.get('critical_issues', [])
    if critical_issues:
        st.subheader("⚠️ Validation Issues")
        for issue in critical_issues:
            st.warning(f"• {issue}")

    # Dimensional analysis
    dimensional = validation_data.get('dimensional_analysis', {})
    if dimensional:
        st.subheader("📊 Dimensional Analysis")

        dim_data = []
        for dim_name, dim_result in dimensional.items():
            if hasattr(dim_result, 'confidence_score'):  # Handle dataclass objects
                dim_data.append({
                    'Dimension': dim_name.replace('_', ' ').title(),
                    'Confidence': f"{dim_result.confidence_score:.2f}",
                    'Reliability': dim_result.reliability_level.title(),
                    'Issues': len(dim_result.issues),
                    'Summary': dim_result.summary[:50] + '...' if len(dim_result.summary) > 50 else dim_result.summary
                })
            elif isinstance(dim_result, dict):  # Handle dict objects
                dim_data.append({
                    'Dimension': dim_name.replace('_', ' ').title(),
                    'Confidence': f"{dim_result.get('confidence_score', 0):.2f}",
                    'Reliability': dim_result.get('reliability_level', 'Unknown').title(),
                    'Issues': len(dim_result.get('issues', [])),
                    'Summary': dim_result.get('summary', 'N/A')[:50] + '...' if len(dim_result.get('summary', '')) > 50 else dim_result.get('summary', 'N/A')
                })

        if dim_data:
            st.dataframe(dim_data, use_container_width=True, height=250)

    # Display judge assessments if available
    if 'judge_assessments' in validation_data:
        st.write("---")  # Separator
        st.subheader("🤖 Judge LLM Assessments")
        display_judge_assessments(validation_data['judge_assessments'])

    # Display judge consensus if available
    if 'judge_consensus' in validation_data:
        st.write("---")  # Separator
        st.subheader("📊 Judge Consensus Analysis")
        display_judge_consensus(validation_data['judge_consensus'])


def display_judge_assessments(judge_assessments):
    """Display independent quality assessments from judge LLMs."""
    if not judge_assessments:
        st.info("No judge assessments available")
        return

    # Create tabs for each judge
    judge_names = list(judge_assessments.keys())
    if len(judge_names) == 1:
        # Single judge - no tabs needed
        judge_name = judge_names[0]
        st.write(f"**{judge_name.replace('_', ' ').title()} Assessment:**")
        display_single_judge_assessment(judge_assessments[judge_name])
    else:
        # Multiple judges - use tabs
        tabs = st.tabs([judge_name.replace('_', ' ').title() for judge_name in judge_names])

        for tab, judge_name in zip(tabs, judge_names):
            with tab:
                display_single_judge_assessment(judge_assessments[judge_name])


def display_single_judge_assessment(assessment):
    """Display a single judge's quality assessment."""
    if "error" in assessment:
        st.error(f"Judge assessment failed: {assessment['error']}")
        return

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        score = assessment.get('overall_quality_score', 0)
        st.metric("Quality Score", f"{score}/10")

    with col2:
        suitable = assessment.get('suitable_for_extraction', False)
        st.metric("Suitable for Extraction", "✅ Yes" if suitable else "❌ No")

    with col3:
        judge_name = assessment.get('judge_name', 'Unknown')
        st.metric("Judge", judge_name.replace('_', ' ').title())

    # Quality issues summary
    quality_issues = [
        ('blur_detection', '🌫️ Blur'),
        ('contrast_assessment', '🌗 Contrast'),
        ('glare_identification', '💡 Glare'),
        ('water_stains', '💧 Water Stains'),
        ('tears_or_folds', '📄 Tears/Folds'),
        ('cut_off_detection', '✂️ Cut-off'),
        ('missing_sections', '🔍 Missing Sections'),
        ('obstructions', '🚫 Obstructions')
    ]

    # Create summary table
    issues_data = []
    for issue_key, issue_name in quality_issues:
        if issue_key in assessment and isinstance(assessment[issue_key], dict):
            issue_data = assessment[issue_key]
            issues_data.append({
                'Issue': issue_name,
                'Detected': '✅ Yes' if issue_data.get('detected', False) else '❌ No',
                'Severity': issue_data.get('severity_level', 'N/A').title(),
                'Confidence': f"{issue_data.get('confidence_score', 0):.2f}",
                'Measure': f"{issue_data.get('quantitative_measure', 0):.2f}"
            })

    if issues_data:
        st.dataframe(issues_data, use_container_width=True, height=300)


def display_judge_consensus(consensus):
    """Display consensus analysis from multiple judge assessments."""
    if not consensus or "error" in consensus:
        st.info("No consensus data available")
        return

    # Overall consensus metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        judge_count = consensus.get('judge_count', 0)
        st.metric("Judge Count", judge_count)

    with col2:
        score_consensus = consensus.get('score_consensus', {})
        avg_score = score_consensus.get('average_score', 0)
        st.metric("Average Score", f"{avg_score:.1f}/10")

    with col3:
        score_agreement = score_consensus.get('score_agreement', 'unknown')
        color = "🟢" if score_agreement == "high" else "🟡" if score_agreement == "medium" else "🔴"
        st.metric("Score Agreement", f"{color} {score_agreement.title()}")

    with col4:
        overall_agreement = consensus.get('overall_agreement', 'unknown')
        color = "🟢" if overall_agreement == "high" else "🟡" if overall_agreement == "medium" else "🔴"
        st.metric("Overall Agreement", f"{color} {overall_agreement.title()}")

    # Suitability consensus
    suitability = consensus.get('suitability_consensus', {})
    suitable_pct = suitability.get('suitable_percentage', 0)
    unanimous = suitability.get('unanimous', False)

    st.write(f"**Extraction Suitability:** {suitable_pct:.0f}% of judges agree it's suitable")
    if unanimous:
        st.success("✅ Unanimous agreement on suitability")
    else:
        st.warning("⚠️ Judges disagree on extraction suitability")

    # Issue-level consensus
    issue_consensus = consensus.get('issue_level_consensus', {})
    if issue_consensus:
        st.subheader("📋 Issue-Level Consensus")

        consensus_data = []
        for issue_key, issue_data in issue_consensus.items():
            issue_name = issue_key.replace('_', ' ').title()
            detection_consensus = issue_data.get('detection_consensus', 0)
            severity_agreement = issue_data.get('severity_agreement', False)
            most_common_severity = issue_data.get('most_common_severity', 'N/A')
            avg_confidence = issue_data.get('avg_confidence', 0)

            consensus_data.append({
                'Issue': issue_name,
                'Detection Agreement': f"{detection_consensus*100:.0f}%",
                'Severity Agreement': "✅ Yes" if severity_agreement else "❌ No",
                'Common Severity': most_common_severity.title(),
                'Avg Confidence': f"{avg_confidence:.2f}"
            })

        if consensus_data:
            st.dataframe(consensus_data, use_container_width=True, height=300)


def display_validation_result(validation):
    """Display UQLM validation results in a formatted way."""
    if not validation:
        st.info("No UQLM validation data available")
        return

    st.subheader("🎯 UQLM Validation Results")

    # Check if this is the new readable format
    if 'validation_report' in validation:
        # New readable format
        report = validation['validation_report']
        overall = report.get('overall_assessment', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            confidence = overall.get('confidence_score', 0)
            st.metric("Overall Confidence", f"{confidence:.2f}")

        with col2:
            reliability = overall.get('reliability_level', 'Unknown')
            color = "🟢" if reliability == "HIGH" else "🟡" if reliability == "MEDIUM" else "🔴"
            st.metric("Reliability", f"{color} {reliability}")

        with col3:
            is_reliable = overall.get('is_reliable', False)
            st.metric("Is Reliable", "✅ Yes" if is_reliable else "❌ No")

        with col4:
            issues_count = report.get('critical_issues_summary', {}).get('total_issues', 0)
            st.metric("Critical Issues", issues_count)

        # Recommendation
        if overall.get('recommendation'):
            st.info(f"**Recommendation:** {overall['recommendation']}")

        # Critical Issues
        critical_issues = report.get('critical_issues_summary', {}).get('issues', [])
        if critical_issues:
            st.warning("**Critical Issues Found:**")
            for issue in critical_issues[:5]:  # Show top 5
                st.write(f"• {issue}")



        # Detailed Analysis with Tabs
        detailed = validation.get('detailed_analysis', {})
        if detailed:
            st.subheader("🔍 Detailed UQLM Analysis")

            # Metadata section
            metadata = detailed.get('metadata', {})
            if metadata:
                with st.expander("📊 Validation Metadata"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Country:** {metadata.get('country', 'Unknown')}")
                        st.write(f"**Receipt Type:** {metadata.get('receipt_type', 'Unknown')}")
                        st.write(f"**ICP:** {metadata.get('icp', 'Unknown')}")
                    with col2:
                        st.write(f"**Validation Method:** {metadata.get('validation_method', 'Unknown')}")
                        st.write(f"**Panel Judges:** {metadata.get('panel_judges', 0)}")
                        st.write(f"**Original Issues:** {metadata.get('original_issues_found', 0)}")

            # Dimension Details with Tabs
            dimension_details = detailed.get('dimension_details', {})
            if dimension_details:
                # Create tabs for each validation dimension
                dimension_names = list(dimension_details.keys())
                if dimension_names:
                    # Create user-friendly tab names
                    tab_names = []
                    for dim in dimension_names:
                        friendly_name = dim.replace('_', ' ').title()
                        # Add emojis for better UX
                        emoji_map = {
                            'Factual Grounding': '📋',
                            'Knowledge Base Adherence': '📚',
                            'Compliance Accuracy': '⚖️',
                            'Issue Categorization': '🏷️',
                            'Recommendation Validity': '💡',
                            'Hallucination Detection': '🔍'
                        }
                        emoji = emoji_map.get(friendly_name, '📊')
                        tab_names.append(f"{emoji} {friendly_name}")

                    # Create tabs
                    tabs = st.tabs(tab_names)

                    # Display content for each tab
                    for tab, dim_key in zip(tabs, dimension_names):
                        with tab:
                            details = dimension_details[dim_key]
                            if isinstance(details, dict):
                                # Metrics row
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    confidence = details.get('confidence_score', 0)
                                    st.metric("Confidence Score", f"{confidence:.2f}")
                                with col2:
                                    reliability = details.get('reliability_level', 'unknown').upper()
                                    color = "🟢" if reliability == "HIGH" else "🟡" if reliability == "MEDIUM" else "🔴"
                                    st.metric("Reliability", f"{color} {reliability}")
                                with col3:
                                    issues_count = details.get('total_issues', len(details.get('issues_found', [])))
                                    st.metric("Issues Found", issues_count)

                                # Summary
                                summary = details.get('summary', 'No summary available')
                                if summary:
                                    st.subheader("📝 Analysis Summary")
                                    st.write(summary)

                                # Issues
                                issues = details.get('issues_found', [])
                                if issues:
                                    st.subheader("⚠️ Issues Identified")
                                    for i, issue in enumerate(issues, 1):
                                        st.write(f"**{i}.** {issue}")

                                # Raw response (optional, in expander)
                                raw_response = details.get('raw_response', '')
                                if raw_response and len(raw_response) > 100:
                                    with st.expander("🔍 Detailed Analysis Response"):
                                        st.text_area("Raw Analysis", raw_response, height=200, disabled=True)
                            else:
                                st.write(str(details))

    else:
        # Legacy format fallback
        summary = validation.get('validation_summary', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            confidence = summary.get('overall_confidence', 0)
            st.metric("Overall Confidence", f"{confidence:.2f}")

        with col2:
            reliability = summary.get('reliability_level', 'Unknown')
            color = "🟢" if reliability == "HIGH" else "🟡" if reliability == "MEDIUM" else "🔴"
            st.metric("Reliability", f"{color} {reliability}")

        with col3:
            is_reliable = summary.get('is_reliable', False)
            st.metric("Is Reliable", "✅ Yes" if is_reliable else "❌ No")

        with col4:
            issues_count = summary.get('validated_issues_count', 0)
            st.metric("Validated Issues", issues_count)

        # Recommendation
        if summary.get('recommendation'):
            st.info(f"**Recommendation:** {summary['recommendation']}")

        # Critical Issues
        if summary.get('critical_issues'):
            st.warning("**Critical Issues Found:**")
            for issue in summary['critical_issues'][:5]:  # Show top 5
                st.write(f"• {issue}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💰 Expense Processing System Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    available_countries = get_available_countries()
    country = st.sidebar.selectbox(
        "Country",
        available_countries,
        help="Select the country for compliance rules"
    )

    # Show available compliance data info
    if len(available_countries) > 1:
        st.sidebar.success(f"✅ {len(available_countries)} countries available")
    else:
        st.sidebar.info("ℹ️ Add more JSON files to data/ directory for additional countries")
    
    icp = st.sidebar.selectbox(
        "ICP (Internal Control Provider)",
        ["Global People", "goGlobal", "Parakar", "Atlas"],
        help="Select your ICP for compliance validation"
    )
    
    # Get LlamaIndex API key from environment or user input
    env_api_key = os.getenv("LLAMAPARSE_API_KEY", "")

    if env_api_key:
        st.sidebar.success("✅ LlamaIndex API key loaded from environment")
        llamaparse_api_key = env_api_key
    else:
        st.sidebar.warning("⚠️ LLAMAPARSE_API_KEY not found in environment")
        llamaparse_api_key = st.sidebar.text_input(
            "LlamaIndex API Key",
            type="password",
            value="",
            help="Enter your LlamaIndex API key for document parsing (or set LLAMAPARSE_API_KEY environment variable)"
        )
    
    # Main content
    st.header("📁 Upload Expense Documents")
    
    uploaded_files = st.file_uploader(
        "Choose expense files",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'doc', 'docx'],
        accept_multiple_files=True,
        help="Upload your expense documents (receipts, invoices, etc.)"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) uploaded successfully!")

        # Analyze image quality for uploaded files
        st.subheader("📸 Image Quality Analysis")

        # Store quality results in session state
        if 'image_quality_results' not in st.session_state:
            st.session_state.image_quality_results = {}

        # Analyze each uploaded file
        quality_progress = st.progress(0)
        quality_status = st.empty()

        for i, file in enumerate(uploaded_files):
            if file.type.startswith('image/'):
                quality_status.text(f"Analyzing image quality: {file.name}")

                # Check if we already analyzed this file
                file_key = f"{file.name}_{file.size}"
                if file_key not in st.session_state.image_quality_results:
                    quality_result = analyze_uploaded_file_quality(file)
                    st.session_state.image_quality_results[file_key] = quality_result

                quality_progress.progress((i + 1) / len(uploaded_files))

        quality_status.text("✅ Image quality analysis complete!")
        time.sleep(0.5)  # Brief pause to show completion
        quality_status.empty()
        quality_progress.empty()

        # Add save functionality for quality results
        if st.session_state.image_quality_results:
            st.subheader("💾 Save Quality Assessment Results")

            # Show summary of available results
            num_results = len(st.session_state.image_quality_results)
            successful_results = [r for r in st.session_state.image_quality_results.values() if 'error' not in r]

            st.info(f"📊 **Available Results:** {num_results} file(s) analyzed, {len(successful_results)} successful assessments")

            # Single save button - saves to quality_reports/ directory (same as main.py)
            _, col_center, _ = st.columns([1, 1, 1])
            with col_center:
                if st.button("💾 Save Quality Results", help="Save quality results to quality_reports/ directory", use_container_width=True):
                    with st.spinner("Saving quality results..."):
                        saved_files = save_quality_results_to_files(st.session_state.image_quality_results)
                        if saved_files:
                            st.success(f"✅ Saved {len(saved_files)} quality report(s) to quality_reports/ directory")
                            with st.expander("📄 Saved Files"):
                                for file_path in saved_files:
                                    st.text(f"• {file_path.name}")
                        else:
                            st.error("❌ Failed to save quality results")

            # Add clear results option
            st.markdown("---")
            _, col_clear_center, _ = st.columns([1, 1, 1])
            with col_clear_center:
                if st.button("🗑️ Clear Quality Results", help="Clear all quality assessment results from session"):
                    st.session_state.image_quality_results = {}
                    st.success("✅ Quality results cleared!")
                    st.experimental_rerun()

        # Display uploaded files with previews and quality results
        if len(uploaded_files) == 1:
            # Single file - show quality analysis (image preview is now included in quality display)
            st.subheader("📋 File Analysis")

            if uploaded_files[0].type.startswith('image/'):
                file_key = f"{uploaded_files[0].name}_{uploaded_files[0].size}"
                quality_result = st.session_state.image_quality_results.get(file_key, {})
                display_image_quality_result(quality_result, uploaded_files[0])
            else:
                # For non-image files, show file preview
                st.write("**File Preview:**")
                display_file_preview(uploaded_files[0])
                st.info("Image quality analysis only available for image files")
        else:
            # Multiple files - show in tabs or expanders
            st.subheader("📋 File Previews & Quality Analysis")

            # Create tabs for each file
            if len(uploaded_files) <= 5:  # Use tabs for up to 5 files
                tab_names = [f"📄 {file.name}" for file in uploaded_files]
                tabs = st.tabs(tab_names)

                for tab, file in zip(tabs, uploaded_files):
                    with tab:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.write("**File Preview:**")
                            display_file_preview(file)

                        with col2:
                            if file.type.startswith('image/'):
                                st.write("**Image Quality:**")
                                file_key = f"{file.name}_{file.size}"
                                quality_result = st.session_state.image_quality_results.get(file_key, {})
                                display_image_quality_result(quality_result, file)
                            else:
                                st.info("Image quality analysis only available for image files")
            else:
                # Use expanders for many files
                for i, file in enumerate(uploaded_files):
                    with st.expander(f"📄 {file.name} ({file.size} bytes)", expanded=(i == 0)):
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.write("**File Preview:**")
                            display_file_preview(file)

                        with col2:
                            if file.type.startswith('image/'):
                                st.write("**Image Quality:**")
                                file_key = f"{file.name}_{file.size}"
                                quality_result = st.session_state.image_quality_results.get(file_key, {})
                                display_image_quality_result(quality_result, file)
                            else:
                                st.info("Image quality analysis only available for image files")
        
        # Process button
        if st.button("🚀 Process Expenses", type="primary"):
            if not llamaparse_api_key:
                st.error("Please enter your LlamaIndex API key in the sidebar.")
                return
            
            with st.spinner("Processing your expense documents..."):
                # Run async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results, error = loop.run_until_complete(
                        process_uploaded_files(uploaded_files, country, icp, llamaparse_api_key)
                    )
                    
                    if error:
                        st.error(f"❌ Processing failed: {error}")
                    else:
                        st.session_state.processing_results = results
                        st.session_state.processing_complete = True
                        st.success("✅ Processing completed successfully!")
                        
                finally:
                    loop.close()
    
    # Display results
    if st.session_state.processing_complete and st.session_state.processing_results:
        st.header("📊 Processing Results")
        
        for i, result in enumerate(st.session_state.processing_results):
            file_name = result.get('file_name', f'File {i+1}')
            
            with st.expander(f"📄 {file_name}", expanded=True):
                if result.get('status') == 'completed':
                    
                    # Check if citations are available
                    has_citations = result.get('extraction', {}).get('citations') is not None

                    # Tabs for different result types
                    if has_citations:
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏷️ Classification", "📋 Extraction", "📝 Citations", "⚖️ Compliance", "🎯 UQLM Validation", "📄 Raw JSON"])
                    else:
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏷️ Classification", "📋 Extraction", "⚖️ Compliance", "🎯 UQLM Validation", "📄 Raw JSON"])

                    with tab1:
                        display_classification_result(result.get('classification'))

                    with tab2:
                        display_extraction_result(result.get('extraction'))

                    if has_citations:
                        with tab3:
                            display_citation_result(result.get('extraction', {}).get('citations'))

                        with tab4:
                            display_compliance_result(result.get('compliance'))

                        with tab5:
                            # Load validation results from separate file
                            validation_data = load_validation_result(file_name)
                            display_validation_result(validation_data)

                        with tab6:
                            st.json(result)
                    else:
                        with tab3:
                            display_compliance_result(result.get('compliance'))

                        with tab4:
                            # Load validation results from separate file
                            validation_data = load_validation_result(file_name)
                            display_validation_result(validation_data)

                        with tab5:
                            st.json(result)
                        
                        # Download button for JSON
                        try:
                            # Create a JSON-serializable version of the result
                            serializable_result = make_json_serializable(result)
                            json_str = json.dumps(serializable_result, indent=2)
                        except Exception as e:
                            st.error(f"Error preparing JSON for download: {e}")
                            json_str = json.dumps({"error": "Could not serialize result"}, indent=2)

                        st.download_button(
                            label="💾 Download JSON",
                            data=json_str,
                            file_name=f"{pathlib.Path(file_name).stem}_result.json",
                            mime="application/json"
                        )
                else:
                    st.error(f"❌ Processing failed for {file_name}: {result.get('error', 'Unknown error')}")

        # Add standalone validation section
        st.markdown("---")
        st.header("🎯 LLM-as-Judge Validation")
        st.info("Run additional validation on the processing results using the separated validation system.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Run Standalone Validation", type="secondary", use_container_width=True):
                with st.spinner("Running LLM-as-judge validation..."):
                    # Run async validation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        validation_summary, validation_error = loop.run_until_complete(
                            run_standalone_validation()
                        )

                        if validation_error:
                            st.error(f"❌ Validation failed: {validation_error}")
                        else:
                            st.session_state.validation_results = validation_summary
                            st.success("✅ Standalone validation completed successfully!")

                    finally:
                        loop.close()

        # Display validation results if available
        if hasattr(st.session_state, 'validation_results') and st.session_state.validation_results:
            st.subheader("📋 Validation Summary")

            validation_summary = st.session_state.validation_results

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                total_files = validation_summary.get('total_files_processed', 0)
                st.metric("Files Processed", total_files)

            with col2:
                total_time = validation_summary.get('total_validation_time', 0)
                st.metric("Validation Time", f"{total_time:.1f}s")

            with col3:
                compliance_count = len(validation_summary.get('compliance_validation', {}).get('results', []))
                quality_count = len(validation_summary.get('quality_validation', {}).get('results', []))
                st.metric("Validations", f"{compliance_count + quality_count}")

            # Detailed results
            if validation_summary.get('compliance_validation', {}).get('enabled'):
                st.write("**🔍 Compliance Validation Results:**")
                compliance_results = validation_summary['compliance_validation']['results']

                for result in compliance_results:
                    if result.get('status') == 'completed':
                        confidence = result.get('confidence_score', 0)
                        reliability = result.get('reliability_level', 'unknown')
                        st.success(f"✅ {result['source_file']}: {confidence:.2f} confidence ({reliability} reliability)")
                    else:
                        st.error(f"❌ {result['source_file']}: {result.get('error', 'Unknown error')}")

            if validation_summary.get('quality_validation', {}).get('enabled'):
                st.write("**🤖 Quality Validation Results:**")
                quality_results = validation_summary['quality_validation']['results']

                for result in quality_results:
                    if result.get('status') == 'completed':
                        confidence = result.get('confidence_score', 0)
                        reliability = result.get('reliability_level', 'unknown')
                        st.success(f"✅ {result['source_file']}: {confidence:.2f} confidence ({reliability} reliability)")
                    else:
                        st.error(f"❌ {result['source_file']}: {result.get('error', 'Unknown error')}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💡 <strong>Expense Processing System Demo</strong> - Powered by AI Agents</p>
        <p>Upload your expense documents and see automatic classification, data extraction, and compliance analysis!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
