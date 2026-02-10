#!/usr/bin/env python3
"""
Script to update Pydantic models to follow Pydantic 2.x patterns
"""

import os
import re
from pathlib import Path


def update_pydantic_imports(content: str) -> str:
    """Update Pydantic imports to use v2 patterns"""
    # Update import statements
    content = re.sub(r'from pydantic import ([^,\n]+)', 
                     lambda m: f'from pydantic.v1 import {m.group(1)}' if 'BaseModel' in m.group(1) or 'Field' in m.group(1) else m.group(0), 
                     content)
    
    # Alternative: Use the new Pydantic v2 imports where appropriate
    content = re.sub(r'from pydantic import BaseModel', 'from pydantic import BaseModel', content)
    content = re.sub(r'from pydantic import Field', 'from pydantic import Field', content)
    
    return content


def update_model_usage(content: str) -> str:
    """Update model usage to follow Pydantic 2.x patterns"""
    # Replace dict() with model_dump() for Pydantic 2.x
    content = re.sub(r'\.dict\(\)', '.model_dump()', content)
    
    # Replace schema() with model_json_schema() for Pydantic 2.x
    content = re.sub(r'\.schema\(\)', '.model_json_schema()', content)
    
    # Replace validate() with model_validate() for Pydantic 2.x
    content = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\.validate\(', r'\1.model_validate(', content)
    
    return content


def process_file(file_path: Path):
    """Process a single file to update Pydantic patterns"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply transformations
    content = update_pydantic_imports(content)
    content = update_model_usage(content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated {file_path}")
    else:
        print(f"  No changes needed for {file_path}")


def main():
    """Main function to process all Python files in the project"""
    project_root = Path('.')
    
    # Find all Python files in the project
    py_files = []
    for ext in ['.py']:
        py_files.extend(project_root.rglob(f'*.{ext}'))
    
    # Process each file
    for file_path in py_files:
        if 'venv' not in str(file_path) and '.git' not in str(file_path):
            try:
                process_file(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()