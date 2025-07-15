#!/usr/bin/env python3
"""
Test script to verify data extraction agent changes
"""

import json
from data_extraction_agent import extract_data_from_receipt

def test_data_extraction_with_schema():
    """Test the updated data extraction function."""
    
    # Sample extraction requirements (schema)
    sample_schema = {
        "title": "Test Schema",
        "file_related_requirements": [
            {
                "field_type": "Supplier Name",
                "description": "Name of the supplier/vendor on invoice",
                "receipt_type": "All",
                "mandatory_optional": "Mandatory"
            },
            {
                "field_type": "Total Amount",
                "description": "Total amount on the receipt",
                "receipt_type": "All",
                "mandatory_optional": "Mandatory"
            }
        ]
    }
    
    # Sample receipt text
    sample_receipt = """
    RESTAURANT ABC
    123 Main Street
    Total: $25.50
    Date: 2024-01-15
    """
    
    # Convert schema to JSON string
    schema_json = json.dumps(sample_schema)
    
    print("Testing data extraction with schema...")
    print(f"Schema: {schema_json}")
    print(f"Receipt text: {sample_receipt}")
    
    try:
        # Call the updated function
        result = extract_data_from_receipt(schema_json, sample_receipt)
        
        print("\n✅ Function executed successfully")
        print(f"Result type: {type(result)}")
        
        # Check if result has expected structure
        if isinstance(result, dict):
            if 'extracted_data' in result and 'schema_used' in result:
                print("✅ Result has correct structure with 'extracted_data' and 'schema_used'")
                
                # Check schema_used
                schema_used = result['schema_used']
                print(f"Schema used type: {type(schema_used)}")
                
                if isinstance(schema_used, dict) and 'title' in schema_used:
                    print(f"✅ Schema title: {schema_used['title']}")
                
                # Check extracted_data
                extracted_data = result['extracted_data']
                print(f"Extracted data type: {type(extracted_data)}")
                
                return True
            else:
                print("❌ Result missing expected keys")
                return False
        else:
            print(f"❌ Result is not a dictionary: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return False

if __name__ == "__main__":
    success = test_data_extraction_with_schema()
    if success:
        print("\n✅ Data extraction agent test passed")
    else:
        print("\n❌ Data extraction agent test failed")
