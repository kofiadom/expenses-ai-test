# Professional Expense Processing System

A comprehensive AI-powered expense processing system that automates the complete expense document lifecycle from file extraction to compliance validation.

## 🚀 Features

- **Multi-format Document Processing**: Supports PDFs, images (PNG, JPG, TIFF, etc.)
- **Image Quality Assessment**: Comprehensive quality analysis for image files before processing
- **Intelligent Classification**: Automatically determines if documents are valid expenses
- **Structured Data Extraction**: Extracts comprehensive data including line items, totals, and metadata
- **Compliance Analysis**: Validates against country-specific and ICP-specific policies
- **Concurrent Processing**: Optimized performance with parallel agent execution
- **Professional Architecture**: Built on Agno framework for production reliability

## 🏗️ Architecture

The system uses a multi-agent architecture with the following components:

1. **Image Quality Assessment**: Analyzes image quality (resolution, blur, glare, completeness, damage) before processing
2. **LlamaParse Extractor**: Converts PDFs/images to structured markdown
3. **File Classification Agent**: Determines expense validity and categorization
4. **Data Extraction Agent**: Extracts structured data from documents
5. **Issue Detection Agent**: Analyzes compliance violations and policy issues
6. **Workflow Orchestrator**: Coordinates all agents with state management

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- LlamaIndex API key
- Country-specific compliance data files

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd expense-processing-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
LLAMAINDEX_API_KEY=your_llamaindex_api_key_here
```

4. Prepare your data:
   - Place expense files (PDFs, images) in `expense_files/` directory
   - Ensure compliance data exists in `data/{country}.json`

## 🚀 Usage

### Configuration

Update the configuration variables in `main.py`:

```python
# Configuration - Update these values as needed
COUNTRY = "Germany"
ICP = "Global People"  # Options: "Global People", "goGlobal", "Parakar", "Atlas"
INPUT_FOLDER = "expense_files"
LLAMAPARSE_API_KEY = "your_llamaindex_api_key_here"
DEBUG_MODE = True
```

### Run the System

```bash
python main.py
```

The system will:
1. Process all files in the `expense_files/` directory
2. Automatically classify expense types (no manual input needed)
3. Extract structured data from each document
4. Perform compliance analysis
5. Save individual results to `results/{filename}.json`

## 📊 Output

The system generates individual JSON files for each processed expense document in the `results/` directory:

**Individual Result Files** (`results/{filename}.json`):
- **Classification Result**: Expense type, language, location validation
- **Extraction Result**: Structured data including supplier info, amounts, line items
- **Compliance Result**: Policy violations, recommendations, confidence scores
- **Processing Metadata**: Timestamp, source file, processing status

**Processing Summary**: Console logs with overview of all processed documents

## 🏢 Supported Countries & ICPs

### Countries
- Germany (with comprehensive compliance rules)
- Extensible to other countries by adding compliance JSON files

### ICPs (Internal Company Providers)
- Global People
- goGlobal  
- Parakar
- Atlas

## 📁 Project Structure

```
expense-processing-system/
├── main.py                          # Main entry point
├── expense_processing_workflow.py   # Workflow orchestrator
├── llamaparse_extractor.py         # Document extraction
├── file_classification_agent.py    # Classification agent
├── data_extraction_agent.py        # Data extraction agent
├── issue_detection_agent.py        # Compliance analysis agent
├── data/                           # Compliance data
│   └── germany.json               # German compliance rules
├── expense_files/                  # Input documents
├── results/                        # Output directory
└── requirements.txt               # Dependencies
```

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
LLAMAINDEX_API_KEY=your_llamaindex_api_key
AGNO_TELEMETRY=false  # Optional: disable telemetry
```

### Compliance Data Format

Country-specific compliance data should be in `data/{country}.json` format:

```json
{
  "FileRelatedRequirements": [...],
  "ExpenseTypes": [...]
}
```

## 📸 Image Quality Assessment

The system includes comprehensive image quality assessment for all image files before LlamaParse processing:

### Quality Metrics
- **Resolution Analysis**: DPI calculation and validation against document standards
- **Blur Detection**: Laplacian variance method with motion blur analysis
- **Glare Analysis**: Histogram-based overexposure and bright spot detection
- **Completeness Check**: Edge detection to verify document boundaries and corners
- **Damage Detection**: Identifies stains, tears, and folds in physical documents

### Quality Scoring
- **Overall Score**: 0-100 weighted composite score
- **Pass Threshold**: 70+ considered acceptable for processing
- **Quality Levels**: Excellent (95+), Good (85+), Acceptable (70+), Poor (<70)

### Output Structure
Quality results are saved as separate JSON files in `quality_reports/`:
```json
{
  "overall_assessment": {
    "score": 78.5,
    "level": "Acceptable", 
    "pass_fail": true,
    "issues_summary": ["Slightly blurry", "Missing corner"],
    "recommendations": ["Ensure camera focus", "Include all corners"]
  },
  "score_breakdown": {
    "resolution": {"score": 85.0, "weight": 0.20},
    "blur": {"score": 72.0, "weight": 0.25},
    "glare": {"score": 90.0, "weight": 0.20},
    "completeness": {"score": 65.0, "weight": 0.20},
    "damage": {"score": 80.0, "weight": 0.15}
  }
}
```

### Processing Flow
1. **File Type Detection**: Automatically detects image vs document files
2. **Quality Assessment**: Runs on image files only (PDFs skip this step)
3. **Results Storage**: Quality reports saved to `quality_reports/` directory
4. **Continued Processing**: All files proceed to LlamaParse regardless of quality scores

## 🚨 Error Handling

The system includes comprehensive error handling:

- **File Processing Errors**: Graceful handling of corrupted or unsupported files
- **API Failures**: Retry logic and fallback mechanisms
- **Validation Errors**: Clear error messages for missing data or configuration
- **Compliance Issues**: Detailed reporting of policy violations

## 🔍 Monitoring & Debugging

- Use `--debug` flag for detailed logging
- Check `results/expense_processing.db` for workflow state
- Review individual agent reasoning files for troubleshooting

## 📈 Performance

- **Concurrent Processing**: Classification and extraction run in parallel
- **Caching**: Results cached to avoid reprocessing
- **Optimized Agents**: ~3μs instantiation, ~6.5KB memory footprint
- **Scalable Architecture**: Handles multiple documents efficiently

## 🤝 Contributing

This is a professional PoC system. For production deployment:

1. Add comprehensive test coverage
2. Implement additional country compliance rules
3. Add monitoring and alerting
4. Scale infrastructure for high-volume processing

## 📄 License

Professional PoC - All rights reserved.
