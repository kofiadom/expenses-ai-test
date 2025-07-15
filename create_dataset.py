#!/usr/bin/env python3
"""
Script to create dataset JSON files for expense files
"""

import os
import json
import pathlib

def determine_country(filename):
    """Determine country based on filename patterns"""
    filename_lower = filename.lower()
    
    if 'austrian' in filename_lower or 'austria' in filename_lower:
        return 'Austria'
    elif 'german' in filename_lower or 'germany' in filename_lower:
        return 'Germany'
    elif 'italia' in filename_lower or 'italy' in filename_lower:
        return 'Italy'
    elif 'swiss' in filename_lower or 'switzerland' in filename_lower:
        return 'Switzerland'
    else:
        return 'Unknown'

def create_dataset():
    """Create JSON files for each expense file"""
    expense_files_dir = pathlib.Path('expense_files')
    dataset_dir = pathlib.Path('dataset')
    
    # Ensure dataset directory exists
    dataset_dir.mkdir(exist_ok=True)
    
    # Process each file in expense_files
    for file_path in expense_files_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name
            country = determine_country(filename)
            
            # Create JSON data
            json_data = {
                "filepath": f"expense_files/{filename}",
                "country": country,
                "icp": "Global People"
            }
            
            # Create JSON filename (replace extension with .json)
            json_filename = pathlib.Path(filename).stem + '.json'
            json_path = dataset_dir / json_filename
            
            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Created: {json_path} -> Country: {country}")

if __name__ == "__main__":
    create_dataset()
    print("\nDataset creation completed!")
