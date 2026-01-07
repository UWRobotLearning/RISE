import click
from sim_pipeline.configs.constants import MODEL_DB_PATH
from sim_pipeline.model_manager.model_manager import ModelManager

def get_model(model_name: str, model_identifier: str | None = None) -> str:
    if not MODEL_DB_PATH.exists():
        raise ValueError(f'Model manager not setup. Please run `scripts/setup_model_manager.py`')
    
    with ModelManager(MODEL_DB_PATH) as mm:
        if model_identifier is not None:
            result, query, params = mm.query_models(name=model_name, model_identifier=model_identifier, return_query=True)
        else:
            result, query, params = mm.query_models(name=model_name, return_query=True)
            
        if len(result) == 0:
            raise ValueError(f'No models matching name {model_name} found.')
        elif len(result) > 1:
            mm.visualize_query_result(query, params, add_index=True)
            idx = click.prompt(f'Multiple models matching name {model_name} found. Please select one by index', type=int, default=-1, show_default=False)
        else:
            idx = 0
        model_dir = result[idx].path

    return model_dir