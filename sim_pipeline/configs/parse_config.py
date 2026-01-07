import importlib
from typing import Type
from types import ModuleType
from pathlib import Path
from contextlib import contextmanager
from hydra.core.config_store import ConfigStore
from sim_pipeline.configs.constants import CONFIG_DIR

@contextmanager
def add_to_path(p):
    import sys
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


def find_config_class(module: ModuleType) -> Type:
    md = module.__dict__
    class_list = [md[c] for c in md if (isinstance(md[c], type) and md[c].__module__ == module.__name__)]
    if len(class_list) > 1:
        print(f"WARNING: Multiple classes found in config {module.__name__}. Config invalid")
        return None
    elif len(class_list) == 0:
        print(f"WARNING: No classes found in config {module.__name__}. Config invalid")
        return None
    return class_list[0]


def import_config(module_location: str, abs_path: str) -> Type:
    spec = importlib.util.spec_from_file_location(module_location, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return find_config_class(module)


def recursive_find_config(curr_dir: Path, module_path: list[str], cs: ConfigStore, excluded_names: list[str]):
    for path in curr_dir.iterdir():
        if path.is_file() and path.suffix == '.py' and path.stem not in excluded_names:
            mp = '.'.join(module_path)
            config_class = import_config(f'{mp}.{path.stem}', path)
            if config_class is None:
                continue
            try:
                group = '/'.join([x for x in module_path if x[0] != '_'])
                if group:
                    cs.store(group=group, name=path.stem, node=config_class)
                else:
                    cs.store(name=path.stem, node=config_class)
            except Exception as e:
                print(f"WARNING: Error storing config {path.stem}: {e}. Skipping config.")
                continue
        if path.is_dir() and path.stem not in excluded_names:
            new_module_path = module_path + [path.stem]
            recursive_find_config(path, new_module_path, cs, excluded_names)


def parse_config(config_path=None, exclude=('constants',)):
    if config_path is None:
        config_path = CONFIG_DIR
    this_file = Path(__file__).stem

    with add_to_path(str(config_path)):
        cs = ConfigStore.instance()
        
        excluded = [*exclude, this_file, '__init__', '__pycache__']
        recursive_find_config(config_path, [], cs, excluded)