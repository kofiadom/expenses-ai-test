# Single File Extraction and Compliance Test Script

This script allows you to test extraction and compliance analysis on a single file with configurable parameters.

## 📋 Purpose

- Test extraction and compliance on **one specific file** instead of processing all files in a directory
- Easily configure filename, country, and ICP parameters
- Get detailed results for debugging and analysis
- Perfect for testing specific edge cases or problematic files

## 🚀 Quick Start

1. **Configure the script** by editing the placeholders in `test_single_file_extraction.py`:

```python
# ============================================================================
# CONFIGURATION PLACEHOLDERS - MODIFY THESE VALUES
# ============================================================================

FILENAME = "switzerland"  # File name without extension
COUNTRY = "Switzerland"   # Country for compliance rules
ICP = "Global People"     # ICP name

# Optional: Override default directories
INPUT_FOLDER = "expense_files"      # Directory containing the original file
RESULTS_DIR = "single_file_results" # Directory to save results
```

2. **Run the script**:

```bash
python test_single_file_extraction.py
```

## 📝 Configuration Options

### Required Parameters

- **FILENAME**: Name of the file to test (without extension)
  - Example: `"switzerland"`, `"german_file_2"`, `"receipt_001"`
  - The script will automatically find files with common extensions (.pdf, .jpg, .png, etc.)

- **COUNTRY**: Country for compliance rules
  - Example: `"Germany"`, `"Switzerland"`, `"Austria"`
  - Must match the compliance data files in the `data/` directory

- **ICP**: ICP (Immigration Compliance Partner) name
  - Options: `"Global People"`, `"goGlobal"`, `"Parakar"`, `"Atlas"`

### Optional Parameters

- **INPUT_FOLDER**: Directory containing the source file (default: `"expense_files"`)
- **RESULTS_DIR**: Directory to save results (default: `"single_file_results"`)

## 📊 What the Script Does

1. **File Discovery**: Finds your source file with various extensions
2. **LlamaParse Extraction**: Converts the file to markdown
3. **Classification**: Determines if it's an expense and categorizes it
4. **Data Extraction**: Extracts structured data from the document
5. **Compliance Analysis**: Checks for compliance issues and violations
6. **Results Saving**: Saves detailed results in JSON format

## 📁 Output Files

The script creates several output files in the results directory:

- `{filename}_result.json` - Complete processing results
- `{filename}_classification.json` - Classification results only
- `{filename}_extraction.json` - Extraction results only
- `{filename}_compliance.json` - Compliance analysis results only
- `markdown_output/{filename}.md` - Extracted markdown content

## 🔍 Example Usage

### Test a Swiss receipt:
```python
FILENAME = "switzerland"
COUNTRY = "Switzerland"
ICP = "Global People"
```

### Test a German expense:
```python
FILENAME = "german_receipt_001"
COUNTRY = "Germany"
ICP = "goGlobal"
```

### Test with custom directories:
```python
FILENAME = "test_receipt"
COUNTRY = "Austria"
ICP = "Parakar"
INPUT_FOLDER = "test_files"
RESULTS_DIR = "debug_results"
```

## 📋 Requirements

- All the same dependencies as the main workflow
- `LLAMAPARSE_API_KEY` environment variable set
- Source file must exist in the specified input folder
- Compliance data file must exist for the specified country

## 🐛 Troubleshooting

### File Not Found
- Check that the file exists in the input folder
- Verify the filename is correct (without extension)
- The script searches for common extensions automatically

### Missing Compliance Data
- Ensure `data/{country}.json` exists (e.g., `data/switzerland.json`)
- Check that the country name matches exactly

### API Key Issues
- Verify `LLAMAPARSE_API_KEY` is set in your environment variables
- Check your `.env` file

## 💡 Tips

- Use this script to test problematic files before running the full workflow
- Great for debugging extraction or compliance issues
- Results are saved individually for easy analysis
- Check the console output for detailed processing logs
