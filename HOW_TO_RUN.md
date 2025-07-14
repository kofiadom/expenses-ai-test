# How to Run the Expense Processing System

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **OpenAI API Key** (set as environment variable)
3. **LlamaIndex API Key** (for document parsing)

## 🔧 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Configure the System
Edit `main.py` and update the configuration variables:
```python
# Configuration - Update these values as needed
COUNTRY = "Germany"                    # Country for compliance rules
ICP = "Global People"                  # ICP name: "Global People", "goGlobal", "Parakar", "Atlas"
INPUT_FOLDER = "expense_files"         # Directory containing expense files
LLAMAPARSE_API_KEY = "your_llamaindex_api_key_here"
DEBUG_MODE = True                      # Set to False for production
```

## 📁 Directory Structure

```
expense-processing-system/
├── main.py                          # Main entry point
├── expense_files/                   # PUT YOUR FILES HERE
│   ├── receipt1.pdf                # Your expense documents
│   ├── invoice2.jpg                # Supported: PDF, DOC, DOCX, PNG, JPG, JPEG, TIFF, BMP, GIF
│   └── hotel_bill3.png
├── results/                         # Generated results (auto-created)
│   ├── receipt1.json               # Individual processing results
│   ├── invoice2.json
│   └── hotel_bill3.json
├── data/
│   └── germany.json               # Compliance rules (add more countries as needed)
└── ...
```

## 🚀 How to Run

### Step 1: Add Your Files
Place your expense documents in the `expense_files/` directory:
```bash
# Copy your files to the expense_files directory
cp /path/to/your/receipts/* expense_files/
```

### Step 2: Run the System
```bash
python main.py
```

### Step 3: Check Results
After processing completes, check the `results/` directory for individual JSON files:
```bash
ls results/
# Output: receipt1.json  invoice2.json  hotel_bill3.json
```

## 📊 Understanding the Output

Each processed file generates a comprehensive JSON result:

```json
{
  "source_file": "hotel_receipt.pdf",
  "processing_timestamp": "2025-07-11T21:37:09.040133",
  "classification_result": {
    "is_expense": true,
    "expense_type": "accommodation",    // Automatically determined!
    "language": "German",
    "language_confidence": 98,
    "document_location": "Germany",
    "location_match": true,
    "classification_confidence": 95
  },
  "extraction_result": {
    "supplier_name": "Hotel Berlin",
    "supplier_address": "Unter den Linden 77, 10117 Berlin, Germany",
    "total_amount": 260.02,
    "currency": "EUR",
    "date_of_issue": "2024-01-15",
    "line_items": [...]
  },
  "compliance_result": {
    "validation_result": {
      "is_valid": false,
      "confidence_score": 0.95,
      "issues_count": 3,
      "issues": [
        {
          "issue_type": "Standards & Compliance | Fix Identified",
          "field": "vat_number",
          "description": "Missing VAT number",
          "recommendation": "Ensure receipt includes VAT number"
        }
      ]
    }
  },
  "processing_status": "completed"
}
```

## 🔍 Key Features

- **Automatic Classification**: No need to specify receipt type - the system determines it automatically
- **Multi-format Support**: Handles PDFs, images, and documents
- **Comprehensive Analysis**: Classification → Extraction → Compliance checking
- **Individual Results**: One JSON file per processed document
- **Detailed Logging**: Full visibility into processing steps

## 🐛 Troubleshooting

### Common Issues:

1. **"Missing OPENAI_API_KEY"**
   - Set the environment variable in `.env` file

2. **"Input folder not found"**
   - Create the `expense_files/` directory and add your files

3. **"No supported files found"**
   - Ensure your files have supported extensions: PDF, DOC, DOCX, PNG, JPG, JPEG, TIFF, BMP, GIF

4. **"Compliance data not found"**
   - Ensure `data/germany.json` exists (or create compliance data for your country)

### Debug Mode:
Set `DEBUG_MODE = True` in `main.py` for detailed logging.

## 📝 Example Workflow

```bash
# 1. Add files
cp my_receipts/* expense_files/

# 2. Run processing
python main.py

# 3. Check logs (real-time)
# INFO Starting expense processing workflow
# INFO Country: Germany, ICP: Global People
# INFO Extracting documents from expense_files using LlamaParse
# INFO Successfully extracted 3 documents
# INFO Processing document 1/3: receipt1.md
# INFO Classification completed for receipt1.md
# INFO Data extraction completed for receipt1.md
# INFO Starting compliance analysis for receipt1.md (type: meals)
# INFO Compliance analysis completed for receipt1.md
# INFO Saved individual result: results/receipt1.json
# ...

# 4. View results
cat results/receipt1.json
```

That's it! The system will automatically:
- Extract text from your documents
- Classify the expense type
- Extract structured data
- Perform compliance analysis
- Save comprehensive results

No manual input required - just add files and run!
