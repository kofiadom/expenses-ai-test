#!/usr/bin/env python3
"""
Professional Expense Processing System - Main Entry Point
"""

import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv
from agno.utils.log import logger
from agno.storage.sqlite import SqliteStorage

from expense_processing_workflow import ExpenseProcessingWorkflow

# Load environment variables
load_dotenv()

# Configuration - Update these values as needed
DATASET_DIR = "dataset"
INPUT_FOLDER = "expense_files"
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
DEBUG_MODE = True

def validate_setup() -> bool:
    """Validate environment and configuration."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Missing OPENAI_API_KEY environment variable")
        return False

    input_path = pathlib.Path(INPUT_FOLDER)
    if not input_path.exists():
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        return False

    dataset_path = pathlib.Path(DATASET_DIR)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        return False

    if not LLAMAPARSE_API_KEY:
        logger.error("LlamaIndex API key is required")
        return False

    return True

async def main():
    """Main entry point for the expense processing system."""

    if not validate_setup():
        sys.exit(1)

    if DEBUG_MODE:
        import logging
        logging.getLogger("agno").setLevel(logging.DEBUG)

    logger.info("Starting dataset-based expense processing workflow")
    logger.info(f"Dataset directory: {DATASET_DIR}")
    logger.info(f"Input folder: {INPUT_FOLDER}")

    try:
        workflow = ExpenseProcessingWorkflow(
            session_id=f"expense-processing-dataset-{DATASET_DIR}",
            storage=SqliteStorage(
                table_name="expense_processing_workflows",
                db_file="expense_processing.db"
            ),
            debug_mode=DEBUG_MODE
        )

        async for response in workflow.process_expenses(
            dataset_dir=DATASET_DIR,
            llamaparse_api_key=LLAMAPARSE_API_KEY,
            input_folder=INPUT_FOLDER
        ):
            logger.info(f"Workflow update: {response.content}")

        summary = workflow.session_state.get("summary", "No summary available")
        logger.info(f"Processing completed. {summary}")
        logger.info("Individual results saved to results/ directory")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
