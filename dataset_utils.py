#!/usr/bin/env python3
"""
Dataset utilities for expense processing system
"""

import json
import pathlib
from typing import List, Dict
from agno.utils.log import logger

def load_dataset_entries(dataset_dir: str = "dataset") -> List[Dict]:
    """
    Load all dataset entries from the dataset directory.
    
    Args:
        dataset_dir: Directory containing dataset JSON files
        
    Returns:
        List of dataset entries with metadata
    """
    dataset_path = pathlib.Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return []
    
    entries = []
    
    for json_file in dataset_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                entry['dataset_file'] = json_file.name
                entries.append(entry)
                logger.debug(f"Loaded dataset entry: {json_file.name}")
        except Exception as e:
            logger.error(f"Failed to load dataset file {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(entries)} dataset entries from {dataset_dir}")
    return entries

def validate_dataset_entry(entry: Dict) -> bool:
    """
    Validate that a dataset entry has required fields.
    
    Args:
        entry: Dataset entry to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['filepath', 'country', 'icp']
    
    for field in required_fields:
        if field not in entry:
            logger.error(f"Missing required field '{field}' in dataset entry")
            return False
        
        if not entry[field] or not isinstance(entry[field], str):
            logger.error(f"Invalid value for field '{field}' in dataset entry")
            return False
    
    # Check if the file actually exists
    filepath = pathlib.Path(entry['filepath'])
    if not filepath.exists():
        logger.warning(f"File not found: {entry['filepath']}")
        return False
    
    return True

def get_dataset_summary(entries: List[Dict]) -> Dict:
    """
    Generate a summary of the dataset entries.
    
    Args:
        entries: List of dataset entries
        
    Returns:
        Summary statistics
    """
    if not entries:
        return {"total_files": 0, "countries": [], "icps": []}
    
    countries = set()
    icps = set()
    file_types = {}
    
    for entry in entries:
        if validate_dataset_entry(entry):
            countries.add(entry['country'])
            icps.add(entry['icp'])
            
            # Count file types
            filepath = pathlib.Path(entry['filepath'])
            ext = filepath.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    return {
        "total_files": len(entries),
        "valid_files": len([e for e in entries if validate_dataset_entry(e)]),
        "countries": sorted(list(countries)),
        "icps": sorted(list(icps)),
        "file_types": file_types
    }
