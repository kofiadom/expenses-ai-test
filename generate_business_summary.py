#!/usr/bin/env python3

import json
import pandas as pd
import os
from datetime import datetime

def load_json_file(filepath):
    """Load JSON file if it exists, return None otherwise"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def get_file_list():
    """Get list of all unique file names from dataset directory"""
    files = set()
    
    if os.path.exists('dataset'):
        for filename in os.listdir('dataset'):
            if filename.endswith('.json'):
                base_name = filename.replace('.json', '')
                if base_name != 'summary_report':
                    files.add(base_name)
    
    return sorted(list(files))

def extract_business_data(file_base):
    """Extract business-relevant data for a single file"""
    
    # Initialize row data
    row_data = {'filename': file_base}
    
    # Load data from each source with fallback patterns
    results_data = load_json_file(f'results/{file_base}.json')
    
    # Try different validation file patterns
    validation_data = (load_json_file(f'validation_results/{file_base}_compliance_validation.json') or 
                      load_json_file(f'validation_results/{file_base}_validation.json'))
    
    # Try different quality file patterns
    quality_data = (load_json_file(f'quality_reports/{file_base}_page1_quality.json') or 
                   load_json_file(f'quality_reports/{file_base}_quality.json'))
    
    dataset_data = load_json_file(f'dataset/{file_base}.json')
    citation_data = load_json_file(f'citation_folder/{file_base}_citation.json')
    
    # Try different LLM quality file patterns
    llm_quality_data = (load_json_file(f'llm_quality_reports/llm_quality_{file_base}_pdf.json') or 
                       load_json_file(f'llm_quality_reports/llm_quality_{file_base}.json'))
    
    # Extract language and isExpense from results
    if results_data and 'classification_result' in results_data:
        classification = results_data['classification_result']
        row_data['language'] = classification.get('language', 'Unknown')
        row_data['isExpense'] = classification.get('is_expense', False)
    else:
        row_data['language'] = 'Unknown'
        row_data['isExpense'] = False
    
    # Extract document location from dataset
    if dataset_data:
        row_data['document_location'] = dataset_data.get('country', 'Unknown')
    else:
        row_data['document_location'] = 'Unknown'
    
    # Extract issue count from results
    if results_data and 'compliance_result' in results_data:
        compliance = results_data['compliance_result']
        validation = compliance.get('validation_result', {})
        row_data['issue_count'] = validation.get('issues_count', 0)
    else:
        row_data['issue_count'] = 0
    
    # Extract validation confidence score
    if validation_data and 'validation_report' in validation_data:
        overall = validation_data['validation_report'].get('overall_assessment', {})
        row_data['validation_confidence_score'] = overall.get('confidence_score', 0)
    else:
        row_data['validation_confidence_score'] = 0
    
    # Extract average confidence from citation
    if citation_data and 'metadata' in citation_data:
        metadata = citation_data['metadata']
        row_data['citation_average_confidence'] = metadata.get('average_confidence', 0)
    else:
        row_data['citation_average_confidence'] = 0
    
    # Extract image quality overall score
    if quality_data and 'overall_assessment' in quality_data:
        overall = quality_data['overall_assessment']
        row_data['image_quality_score'] = overall.get('score', 0)
    else:
        row_data['image_quality_score'] = 0
    
    # Extract LLM quality data
    if llm_quality_data:
        row_data['llm_quality_score'] = llm_quality_data.get('overall_quality_score', 0)
        row_data['llm_suitable_for_extraction'] = llm_quality_data.get('suitable_for_extraction', False)
    else:
        row_data['llm_quality_score'] = 0
        row_data['llm_suitable_for_extraction'] = False
    
    return row_data

def generate_business_summary():
    """Generate business summary report"""
    print("Generating Business Summary Report...")
    
    # Get list of files to process
    file_list = get_file_list()
    print(f"Found {len(file_list)} files to process")
    
    # Extract data for all files
    report_data = []
    for file_base in file_list:
        try:
            row_data = extract_business_data(file_base)
            report_data.append(row_data)
            print(f"Processed: {file_base}")
        except Exception as e:
            print(f"Error processing {file_base}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    
    # Reorder columns for better readability
    column_order = [
        'filename',
        'language',
        'document_location',
        'isExpense',
        'issue_count',
        'validation_confidence_score',
        'citation_average_confidence',
        'image_quality_score',
        'llm_quality_score',
        'llm_suitable_for_extraction'
    ]
    
    # Only include columns that exist
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # Sort by filename for consistency
    df = df.sort_values('filename').reset_index(drop=True)
    
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to CSV
    csv_filename = 'business_summary_report.csv'
    df.to_csv(csv_filename, index=False)
    
    # Generate summary statistics
    print(f"\n=== BUSINESS SUMMARY REPORT ===")
    print(f"Generated: {timestamp}")
    print(f"Total Files Processed: {len(df)}")
    print(f"Output File: {csv_filename}")
    
    print(f"\n=== KEY STATISTICS ===")
    print(f"Expense Documents: {df['isExpense'].sum()} ({df['isExpense'].mean()*100:.1f}%)")
    print(f"Average Issue Count: {df['issue_count'].mean():.2f}")
    print(f"Average Validation Confidence: {df['validation_confidence_score'].mean():.2f}")
    print(f"Average Citation Confidence: {df['citation_average_confidence'].mean():.2f}")
    print(f"Average Image Quality Score: {df['image_quality_score'].mean():.2f}")
    print(f"Average LLM Quality Score: {df['llm_quality_score'].mean():.2f}")
    print(f"LLM Suitable for Extraction: {df['llm_suitable_for_extraction'].sum()} ({df['llm_suitable_for_extraction'].mean()*100:.1f}%)")
    
    print(f"\n=== LANGUAGE DISTRIBUTION ===")
    language_counts = df['language'].value_counts()
    for lang, count in language_counts.items():
        print(f"{lang}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n=== DOCUMENT LOCATION DISTRIBUTION ===")
    location_counts = df['document_location'].value_counts()
    for location, count in location_counts.items():
        print(f"{location}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n=== QUALITY ASSESSMENT ===")
    high_quality_threshold = 0.8
    high_quality_count = (df['image_quality_score'] >= high_quality_threshold).sum()
    print(f"High Quality Images (≥{high_quality_threshold}): {high_quality_count} ({high_quality_count/len(df)*100:.1f}%)")
    
    low_issue_threshold = 2
    low_issue_count = (df['issue_count'] <= low_issue_threshold).sum()
    print(f"Low Issue Count (≤{low_issue_threshold}): {low_issue_count} ({low_issue_count/len(df)*100:.1f}%)")
    
    high_confidence_threshold = 0.8
    high_confidence_count = (df['validation_confidence_score'] >= high_confidence_threshold).sum()
    print(f"High Validation Confidence (≥{high_confidence_threshold}): {high_confidence_count} ({high_confidence_count/len(df)*100:.1f}%)")
    
    return csv_filename

if __name__ == "__main__":
    generate_business_summary()
