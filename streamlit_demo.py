#!/usr/bin/env python3
"""
Streamlit Demo Interface for Expense Processing System
"""

import streamlit as st
import asyncio
import json
import pathlib
import tempfile
import time
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF for PDF preview
from agno.utils.log import logger
from expense_processing_workflow import ExpenseProcessingWorkflow

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
        
        # Save uploaded files to temporary directory
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_path / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            logger.info(f"Saved uploaded file: {uploaded_file.name}")
        
        # Create workflow
        workflow = ExpenseProcessingWorkflow(
            session_id=f"streamlit-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            debug_mode=False
        )
        
        # Process files
        results = []
        progress_placeholder = st.empty()
        
        try:
            async for response in workflow.process_expenses(
                country=country,
                icp=icp,
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
    """Display extraction results in a formatted way."""
    if not extraction:
        st.warning("No extraction data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Supplier Information")
        st.write(f"**Name:** {extraction.get('supplier_name', 'N/A')}")
        st.write(f"**Address:** {extraction.get('supplier_address', 'N/A')}")
        st.write(f"**VAT Number:** {extraction.get('vat_number', 'N/A')}")
    
    with col2:
        st.subheader("Transaction Details")
        st.write(f"**Total Amount:** {extraction.get('currency', '')} {extraction.get('total_amount', 'N/A')}")
        st.write(f"**Date:** {extraction.get('date_of_issue', 'N/A')}")
        st.write(f"**Currency:** {extraction.get('currency', 'N/A')}")
    
    # Line items
    if extraction.get('line_items'):
        st.subheader("Line Items")
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
                "Total": f"{extraction.get('currency', '')} {item.get('total_price', 'N/A')}"
            })
        st.dataframe(line_items_data, use_container_width=True)

def display_compliance_result(compliance):
    """Display compliance results in a formatted way."""
    if not compliance or 'validation_result' not in compliance:
        st.warning("No compliance data available")
        return
    
    validation = compliance['validation_result']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        is_valid = validation.get('is_valid', False)
        st.metric("Compliance Status", "✅ Valid" if is_valid else "❌ Issues Found")
    
    with col2:
        st.metric("Confidence Score", f"{validation.get('confidence_score', 0):.2f}")
    
    with col3:
        st.metric("Issues Count", validation.get('issues_count', 0))
    
    # Display issues
    if validation.get('issues'):
        st.subheader("Compliance Issues")
        for i, issue in enumerate(validation['issues'], 1):
            with st.expander(f"Issue {i}: {issue.get('issue_type', 'Unknown')}"):
                st.write(f"**Field:** {issue.get('field', 'N/A')}")
                st.write(f"**Description:** {issue.get('description', 'N/A')}")
                st.write(f"**Recommendation:** {issue.get('recommendation', 'N/A')}")
    
    if validation.get('compliance_summary'):
        st.text_area("Compliance Summary", validation['compliance_summary'], height=100)

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
    
    llamaparse_api_key = st.sidebar.text_input(
        "LlamaIndex API Key",
        type="password",
        value="llx-2uTsxLAvQ9K64yhww0GLfBwWYSJkwokpHSmCXmWTJ2U9bDH8",
        help="Enter your LlamaIndex API key for document parsing"
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

        # Display uploaded files with previews
        if len(uploaded_files) == 1:
            # Single file - show full preview
            st.subheader("📋 File Preview")
            display_file_preview(uploaded_files[0])
        else:
            # Multiple files - show in tabs or expanders
            st.subheader("📋 File Previews")

            # Create tabs for each file
            if len(uploaded_files) <= 5:  # Use tabs for up to 5 files
                tab_names = [f"📄 {file.name}" for file in uploaded_files]
                tabs = st.tabs(tab_names)

                for tab, file in zip(tabs, uploaded_files):
                    with tab:
                        display_file_preview(file)
            else:
                # Use expanders for many files
                for i, file in enumerate(uploaded_files):
                    with st.expander(f"📄 {file.name} ({file.size} bytes)", expanded=(i == 0)):
                        display_file_preview(file)
        
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
                    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Classification", "📋 Extraction", "⚖️ Compliance", "📄 Raw JSON"])
                    
                    with tab1:
                        display_classification_result(result.get('classification'))
                    
                    with tab2:
                        display_extraction_result(result.get('extraction'))
                    
                    with tab3:
                        display_compliance_result(result.get('compliance'))
                    
                    with tab4:
                        st.json(result)
                        
                        # Download button for JSON
                        json_str = json.dumps(result, indent=2)
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
