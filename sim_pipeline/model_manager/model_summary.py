import json
from pathlib import Path
from sim_pipeline.model_manager.get_model import get_model

def get_model_summary(name: str, model_identifier: str | None, model_dir_path: str | None):
    if model_dir_path is not None:
        model_dir = model_dir_path
    else:
        model_dir = get_model(name, model_identifier)

    model_dir = Path(model_dir)
    metadata_path = model_dir / 'metadata.json'
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print(f'Model Directory: {model_dir}')
    print('==== Model Metadata ====')
    print(f'Name: {metadata["name"]}')
    print(f'Identifier: {metadata["identifier"]}')
    print(f'Policy ID: {metadata["policy_id"]}')
    print(f'Policy Type: {metadata["policy_type"]}')
    print(f'Creator: {metadata["creator"]}')
    print(f'Date: {metadata["date"]}')
    print(f'Command: {metadata["command"]}')
    print(f'Description: {metadata["description"]}')