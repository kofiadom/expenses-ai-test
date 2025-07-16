import json
import pandas as pd
import os
from pathlib import Path

def load_json_file(filepath):
    """Load JSON file if it exists, return None otherwise"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def extract_results_data(data):
    """Extract data from results folder JSON"""
    if not data:
        return {}
    
    # Extract classification data
    classification = data.get('classification_result', {})
    
    # Extract extraction result count (count non-null fields in extraction_result)
    extraction_result = data.get('extraction_result', {})
    extraction_count = sum(1 for v in extraction_result.values() if v is not None and v != "")
    
    # Extract compliance issues count
    compliance = data.get('compliance_result', {})
    validation = compliance.get('validation_result', {})
    issues_count = validation.get('issues_count', 0)
    
    return {
        'extraction_result_count': extraction_count,
        'issues_count': issues_count,
        'is_expense': classification.get('is_expense', False),
        'language': classification.get('language', ''),
        'language_confidence': classification.get('language_confidence', 0),
        'classification_confidence': classification.get('classification_confidence', 0)
    }

def extract_validation_data(data):
    """Extract data from validation_results folder JSON"""
    if not data:
        return {}
    
    validation_report = data.get('validation_report', {})
    overall = validation_report.get('overall_assessment', {})
    dimensional = validation_report.get('dimensional_analysis_summary', {})
    
    return {
        'confidence_score': overall.get('confidence_score', 0),
        'is_reliable': overall.get('is_reliable', False),
        'hallucination_detection': dimensional.get('hallucination_detection', {}).get('confidence', 0),
        'compliance_accuracy': dimensional.get('compliance_accuracy', {}).get('confidence', 0),
        'factual_grounding': dimensional.get('factual_grounding', {}).get('confidence', 0),
        'knowledge_base_adherence': dimensional.get('knowledge_base_adherence', {}).get('confidence', 0)
    }

def extract_quality_data(data):
    """Extract data from quality_test_results folder JSON"""
    if not data:
        return {}
    
    overall = data.get('overall_assessment', {})
    detailed = data.get('detailed_results', {})
    
    return {
        'quality_score': overall.get('score', 0),
        'quality_level': overall.get('level', ''),
        'quality_passed': overall.get('pass_fail', 'False') == 'True',
        'resolution_score': detailed.get('resolution', {}).get('quality', {}).get('score', 0),
        'blur_score': detailed.get('blur', {}).get('metrics', {}).get('blur_score', 0),
        'glare_score': detailed.get('glare', {}).get('glare_analysis', {}).get('glare_score', 0),
        'completeness_score': detailed.get('completeness', {}).get('completeness_score', 0),
        'damage_score': detailed.get('damage', {}).get('damage_score', 0)
    }

def get_file_list():
    """Get list of all unique file names across all directories"""
    files = set()
    
    # Check each directory for files
    directories = ['results', 'validation_results', 'quality_test_results_20250715_142736', 'dataset']
    
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    base_name = filename.replace('.json', '')
                    # Filter out summary_report as it's not an individual file
                    if base_name != 'summary_report':
                        files.add(base_name)
    
    return sorted(list(files))

def generate_csv_report():
    """Generate comprehensive CSV report"""
    
    # Get all unique file names
    file_list = get_file_list()
    print(f"Found {len(file_list)} files to process: {file_list}")
    
    # Initialize data list
    report_data = []
    
    for file_base in file_list:
        print(f"Processing {file_base}...")
        
        # Initialize row data
        row_data = {'file': file_base}
        
        # Load data from each source
        results_data = load_json_file(f'results/{file_base}.json')
        validation_data = load_json_file(f'validation_results/{file_base}.json')
        quality_data = load_json_file(f'quality_test_results_20250715_142736/{file_base}.json')
        dataset_data = load_json_file(f'dataset/{file_base}.json')
        
        # Extract data from each source
        row_data.update(extract_results_data(results_data))
        row_data.update(extract_validation_data(validation_data))
        row_data.update(extract_quality_data(quality_data))
        
        # Add dataset info if available
        if dataset_data:
            row_data['country'] = dataset_data.get('country', '')
            row_data['icp'] = dataset_data.get('icp', '')
        
        report_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Add QA columns for each field (except file)
    qa_columns = {}
    for col in df.columns:
        if col != 'file':
            qa_columns['QA'] = ''
    
    # Add QA columns to dataframe
    for qa_col, default_val in qa_columns.items():
        df[qa_col] = default_val
    
    # Define column order with QA columns interspersed
    columns_order = ['file']
    data_columns = [
        'extraction_result_count',
        'issues_count',
        'is_expense',
        'language',
        'language_confidence',
        'classification_confidence',
        'confidence_score',
        'is_reliable',
        'hallucination_detection',
        'compliance_accuracy',
        'factual_grounding',
        'knowledge_base_adherence',
        'quality_score',
        'quality_level',
        'quality_passed',
        'resolution_score',
        'blur_score',
        'glare_score',
        'completeness_score',
        'damage_score',
        'country',
        'icp'
    ]
    
    # Add each data column followed by QA column
    for col in data_columns:
        if col in df.columns:
            columns_order.append(col)
            columns_order.append('QA')
    
    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    
    # Fill NaN values with appropriate defaults
    df = df.fillna({
        'extraction_result_count': 0,
        'issues_count': 0,
        'is_expense': False,
        'language': '',
        'language_confidence': 0,
        'classification_confidence': 0,
        'confidence_score': 0,
        'is_reliable': False,
        'hallucination_detection': 0,
        'compliance_accuracy': 0,
        'factual_grounding': 0,
        'knowledge_base_adherence': 0,
        'quality_score': 0,
        'quality_level': '',
        'quality_passed': False,
        'resolution_score': 0,
        'blur_score': 0,
        'glare_score': 0,
        'completeness_score': 0,
        'damage_score': 0,
        'country': '',
        'icp': ''
    })
    
    # Save to CSV
    output_file = 'expense_analysis_summary_report.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nCSV report generated successfully: {output_file}")
    print(f"Report contains {len(df)} rows and {len(df.columns)} columns")
    print("\nColumn summary:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Display first few rows
    print(f"\nFirst 3 rows preview:")
    print(df.head(3).to_string())
    
    return output_file

if __name__ == "__main__":
    generate_csv_report()
