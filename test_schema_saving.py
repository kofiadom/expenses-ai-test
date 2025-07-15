#!/usr/bin/env python3
"""
Test script to verify schema saving functionality
"""

import asyncio
import json
import pathlib
import os
from dotenv import load_dotenv
from agno.utils.log import logger

from expense_processing_workflow import ExpenseProcessingWorkflow

# Load environment variables
load_dotenv()

# Configuration for testing
TEST_COUNTRY = "Germany"
TEST_ICP = "Global People"
TEST_INPUT_FOLDER = "expense_files"
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

async def test_schema_saving():
    """Test the schema saving functionality."""
    logger.info("Starting schema saving test")
    
    # Create workflow
    workflow = ExpenseProcessingWorkflow(
        session_id=f"schema-test-{TEST_COUNTRY.lower()}",
        debug_mode=True
    )
    
    # Process a single expense file
    async for response in workflow.process_expenses(
        country=TEST_COUNTRY,
        icp=TEST_ICP,
        llamaparse_api_key=LLAMAPARSE_API_KEY,
        input_folder=TEST_INPUT_FOLDER
    ):
        logger.info(f"Workflow update: {response.content}")
    
    # Check if schema files were created
    schema_dir = pathlib.Path("schemas")
    if not schema_dir.exists():
        logger.error("❌ Schema directory was not created")
        return False
    
    schema_files = list(schema_dir.glob("*.json"))
    if not schema_files:
        logger.error("❌ No schema files were created")
        return False
    
    logger.info(f"✅ Found {len(schema_files)} schema files")
    
    # Verify schema content
    for schema_file in schema_files:
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            logger.info(f"✅ Successfully loaded schema file: {schema_file.name}")
            
            # Check schema structure
            if isinstance(schema_data, dict):
                keys = list(schema_data.keys())
                logger.info(f"Schema keys: {keys}")
                
                # Check for expected sections
                if 'file_related_requirements' in schema_data:
                    requirements = schema_data['file_related_requirements']
                    logger.info(f"Found {len(requirements)} field requirements")
                    
                    # Check first requirement
                    if requirements and isinstance(requirements[0], dict):
                        first_req = requirements[0]
                        logger.info(f"First requirement: {first_req.get('field_type', 'N/A')}")
                else:
                    logger.warning("⚠️ No file_related_requirements found in schema")
            else:
                logger.warning(f"⚠️ Schema is not a dictionary: {type(schema_data)}")
                
        except Exception as e:
            logger.error(f"❌ Error verifying schema file {schema_file.name}: {e}")
            return False
    
    logger.info("✅ Schema saving test completed successfully")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_schema_saving())
        if result:
            print("✅ Schema saving functionality works correctly")
        else:
            print("❌ Schema saving test failed")
    except Exception as e:
        print(f"❌ Test error: {e}")
