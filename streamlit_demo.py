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
    """Analyze the quality of an uploaded image file using the comprehensive quality assessment system."""
    try:
        # Save uploaded file temporarily for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Initialize quality processor
        quality_processor = ImageQualityProcessor(document_type='receipt')

        # Perform comprehensive quality assessment
        quality_results = quality_processor.assess_image_quality(tmp_file_path)

        # Clean up temporary file
        pathlib.Path(tmp_file_path).unlink()

        return quality_results

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
        st.metric("Expense Type", classification.get('expense_type', 'N/A').title())
    
    with col2:
        st.metric("Language", classification.get('language', 'N/A'))
        st.metric("Language Confidence", f"{classification.get('language_confidence', 0)}%")
    
    with col3:
        st.metric("Location Match", "✅ Yes" if classification.get('location_match') else "❌ No")
        st.metric("Classification Confidence", f"{classification.get('classification_confidence', 0)}%")
    
    if classification.get('reasoning'):
        st.text_area("Classification Reasoning", classification['reasoning'], height=100)

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

def display_image_quality_result(image_quality):
    """Display comprehensive image quality analysis results."""
    if not image_quality:
        st.info("No image quality analysis available")
        return

    if "error" in image_quality:
        st.error(f"Image quality analysis failed: {image_quality['error']}")
        return

    st.subheader("📸 Quality Analysis")

    # Overall assessment
    overall = image_quality.get('overall_assessment', {})

    # Main metrics with smaller text
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
        processing_time = image_quality.get('processing_time_seconds', 0)
        st.metric("Time", f"{processing_time:.1f}s")

    # Score breakdown
    score_breakdown = image_quality.get('score_breakdown', {})
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

    # Detailed results in tabs
    detailed = image_quality.get('detailed_results', {})
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

    # Image type detection info
    image_type = image_quality.get('image_type_detection', {})
    if image_type:
        with st.expander("📱 Image Type Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Is Screenshot", "📱 Yes" if image_type.get('is_digital_screenshot') else "📷 No")
                st.metric("Image Subtype", image_type.get('image_subtype', 'Unknown'))
            with col2:
                st.metric("Detection Confidence", f"{image_type.get('confidence', 0)*100:.0f}%")

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
            # Single file - show full preview and quality
            st.subheader("📋 File Preview & Quality Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**File Preview:**")
                display_file_preview(uploaded_files[0])

            with col2:
                if uploaded_files[0].type.startswith('image/'):
                    st.write("**Image Quality:**")
                    file_key = f"{uploaded_files[0].name}_{uploaded_files[0].size}"
                    quality_result = st.session_state.image_quality_results.get(file_key, {})
                    display_image_quality_result(quality_result)
                else:
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
                                display_image_quality_result(quality_result)
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
                                display_image_quality_result(quality_result)
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
                    
                    # Tabs for different result types
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏷️ Classification", "📋 Extraction", "⚖️ Compliance", "🎯 UQLM Validation", "📄 Raw JSON"])

                    with tab1:
                        display_classification_result(result.get('classification'))

                    with tab2:
                        display_extraction_result(result.get('extraction'))

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
