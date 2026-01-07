import sqlite3
import json
import tabulate

from dvc.repo import Repo
from pathlib import Path
from importlib.resources import files
from collections import namedtuple

from sim_pipeline.configs.constants import PolicyType, ROOT_DIR
from sim_pipeline.data_manager.data_manager import _wrap_text_to_colwidths
from sim_pipeline.utils.train_utils import get_latest_diffusion_policy, get_latest_sb_policy, get_latest_robomimic_policy

class ModelManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.commit()
        self.close
        
    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.close()
        
    def create_tables(self):
        """
        Create database tables based on schema. Removes all existing tables.
        """
        self.empty_tables()

        with open(files('sim_pipeline.model_manager') / 'create_tables.sql', 'r') as sql_file:
            sql_script = sql_file.read()
            self.cursor.executescript(sql_script)

            self.populate_initial_types()

            self.conn.commit()
    
    def empty_tables(self):
        """
        Remove all tables from database.
        """
        self.cursor.execute("PRAGMA foreign_keys = OFF;")

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = self.cursor.fetchall()
        
        for table_name in tables:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name[0]};")

        try:
            self.cursor.execute("DELETE FROM sqlite_sequence;")
        except sqlite3.OperationalError:
            pass

        self.cursor.execute("PRAGMA foreign_keys = ON;")
        
    def populate_initial_types(self):
        """
        Initialize the tables with the type definitions of each enum.
        """
        self.cursor.executemany(
            'INSERT INTO PolicyTypes (type) VALUES (?)',
            [(status.value,) for status in PolicyType]
        )
        
    def parse_directories(self, directories: list[Path]):
        """
        Parse directories for model metadata and add to database.
        """
        for dirname in directories:
            if not dirname.is_dir():
                print(f'Warning: {dirname} is not a directory.')
                continue
            
            for path in dirname.iterdir():
                if path.is_file() and path.name == 'metadata.json':
                    print(f'Parsing {path}')
                    self.add_model_metadata(path)
                elif path.is_dir():
                    self.parse_directories([path])

    def add_model_metadata(self, filepath: Path):
        """
        Add model metadata to database.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            if not metadata['valid']:
                return
            
            path = filepath.parent
            
            self.insert_data(
                'Models',
                {
                    'id': str(metadata['policy_id']),
                    'name': metadata['name'],
                    'identifier': metadata['identifier'],
                    'creator': metadata['creator'],
                    'path': str(path),
                    'description': metadata['description'],
                }
            )
            
            policy_type_id = self.fetch_id('PolicyTypes', 'type', metadata['policy_type'])
            self.insert_data(
                'ModelPolicyTypes',
                {
                    'model_id': metadata['policy_id'],
                    'policy_type_id': policy_type_id,
                }
            )
            
            if 'dataset_uuid' in metadata:
                self.insert_data(
                    'ModelDatasetAssociations',
                    {
                        'model_id': metadata['policy_id'],
                        'dataset_id': metadata['dataset_uuid'],
                        'dataset_name': metadata['dataset_name'],
                    }
                )

    def insert_data(self, table_name, data):
        columns = ', '.join(data.keys())
        values = ', '.join([f"'{v}'" for v in data.values()])
        self.cursor.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({values})")


    def fetch_id(self, table_name, matching, name, none_ok=False):
        try:
            return self.cursor.execute(f'SELECT id FROM {table_name} WHERE {matching} = ?', (name,)).fetchone()[0]
        except TypeError:
            if none_ok:
                return None
            raise ValueError(f'{matching} = {name} not found in {table_name} table. Likely {name} was not inserted into enum table.')

    def query_models(self, return_all=False, return_query=False, **kwargs):
        if return_all:
            query_t = namedtuple('query_t', ['id', 'name', 'path', 'identifier', 'creator', 'description', 'policy_type'])
            query = """
            SELECT m.id, m.name, m.path, m.identifier, m.creator, m.description, pt.type
            FROM Models m
            LEFT JOIN ModelPolicyTypes mpt ON m.id = mpt.model_id
            LEFT JOIN PolicyTypes pt ON mpt.policy_type_id = pt.id
            WHERE 1=1
            """
        else:
            query_t = namedtuple('query_t', ['name', 'identifier', 'path'])
            query = """
            SELECT m.id, m.name, m.path
            FROM Models m
            WHERE 1=1
            """
        
        params = []
        
        def add_simple_condition(field, table, column, model_id_column='id'):
            nonlocal query, params
            if field in kwargs:
                values = kwargs[field] if isinstance(kwargs[field], list) else [kwargs[field]]
                placeholders = ','.join(['?' for _ in values])
                query += f" AND m.id IN (SELECT {model_id_column} FROM {table} WHERE {column} IN ({placeholders}))"
                params.extend(values)
                
        def add_subset_condition(field, table, type_column, id_column):
            """
            the values are a subset of the query
            """
            nonlocal query, params
            if field in kwargs:
                values = kwargs[field] if isinstance(kwargs[field], list) else [kwargs[field]]
                placeholders = ','.join(['?' for _ in values])
                query += f"""
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {table}
                        WHERE dataset_id = d.id
                        AND {type_column} NOT IN ({placeholders})
                    )
                    AND EXISTS (
                        SELECT 1
                        FROM {table}
                        WHERE dataset_id = d.id
                    )
                """
                params.extend(values)
        
        add_simple_condition('id', 'Models', 'id')
        add_simple_condition('name', 'Models', 'name')
        add_simple_condition('creator', 'Models', 'creator')
        add_simple_condition('identifier', 'Models', 'identifier')
        
        add_simple_condition('dataset_name', 'ModelDatasetAssociations', 'datatset_name', 'model_id')
        
        add_subset_condition('policy_type', 'ModelPolicyTypes mpt JOIN PolicyTypes pt ON mpt.policy_type_id = pt.id', 'pt.type', 'pt.id')
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        results = [query_t(*result) for result in results]
        if return_query:
            return results, query, params
        return results
    
    def get_all_model_and_metadata_paths(self):
        results = self.query_models(return_all=True)
        paths = [Path(result.path) for result in results]
        policy_types = [PolicyType(result.policy_type) for result in results]
        metadata_paths = [path / 'metadata.json' for path in paths]
        
        all_latest_ckpt_paths = []
        for path, policy_type in zip(paths, policy_types):
            match policy_type:
                case PolicyType.DIFFUSION:
                    ckpt = get_latest_diffusion_policy(path / 'checkpoints')
                case PolicyType.ROBOMIMIC:
                    ckpt = get_latest_robomimic_policy(path / 'models')
                case PolicyType.STABLE_BASELINES:
                    ckpt = get_latest_sb_policy(path / 'policy' / '_step_')
                case _:
                    pass
            all_latest_ckpt_paths.append(ckpt)
        all_latest_ckpt_paths.extend(metadata_paths)
        return all_latest_ckpt_paths
    
    def dvc_add_all(self):
        all_paths = self.get_all_model_and_metadata_paths()
        repo = Repo(ROOT_DIR)
        for path in all_paths:
            repo.add(str(path))
    
    def visualize_table(self, table_name, limit=None):
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [column[1] for column in self.cursor.fetchall()]

        # Get table data
        if limit:
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        else:
            self.cursor.execute(f"SELECT * FROM {table_name};")
        rows = self.cursor.fetchall()

        print(f"\nContents of table '{table_name}':")
        print(tabulate.tabulate(rows, headers=columns, tablefmt="grid"))
        print(f"Total rows: {len(rows)}")

    def visualize_query_result(self, query, params=None, maxcolwidths=25, to_csv=False, add_index=False):
        # Execute the provided query
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # Get column headers from the cursor description
        columns = [column[0] for column in self.cursor.description]
        if add_index:
            columns = ['index'] + columns
            rows = [[i] + list(row) for i, row in enumerate(rows)]

        if to_csv:
            with open('all_models.csv', 'w') as f:
                content = tabulate.tabulate(rows, headers=columns, tablefmt='csv')
                with open('all_models.csv', 'w') as f:
                    f.write(content)
            print(f"\nQuery result saved to all_models.csv")
        else:
            print(f"\nQuery result:")
            print(tabulate.tabulate(rows, headers=columns, tablefmt="grid", maxcolwidths=maxcolwidths))
            print(f"Total rows: {len(rows)}")

    def visualize_all(self, to_csv=False, show_id=False):        
        id_str = 'm.id, ' if show_id else ''
        
        query = f"""
            SELECT DISTINCT {id_str}m.name, m.path, m.identifier, m.creator, pt.type, m.description, mda.dataset_name, mda.dataset_id
            FROM Models m
            LEFT JOIN ModelPolicyTypes mpt ON m.id = mpt.model_id
            LEFT JOIN PolicyTypes pt ON mpt.policy_type_id = pt.id
            LEFT JOIN ModelDatasetAssociations mda ON m.id = mda.model_id
        """
        maxcolwidths = [15, 35, 12, 12, 12, 30, 20, 35]
        maxcolwidths = [50] + maxcolwidths if show_id else maxcolwidths
        self.visualize_query_result(query, maxcolwidths=maxcolwidths, to_csv=to_csv)

tabulate._wrap_text_to_colwidths = _wrap_text_to_colwidths
