import sqlite3
import h5py
import uuid 
import json
import tabulate

from itertools import product
from importlib.resources import files
from typing import Any
from pathlib import Path
from sim_pipeline.configs.constants import COMBINED_DATA_DIR
from sim_pipeline.data_manager.dataset_metadata_enums import *
from sim_pipeline.data_manager.combine_datasets import combine_datasets
from sim_pipeline.reward_functions import RewardFunction
from collections import namedtuple

class DataManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.commit()
        self.close()
        
    def create_tables(self):
        """
        Create database tables based on schema. Removes all existing tables.
        """
        self.empty_tables()

        with open(files('sim_pipeline.data_manager') / 'create_tables.sql', 'r') as sql_file:
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
            'INSERT INTO DatasetProcessedStatuses (category) VALUES (?)',
            [(status.value,) for status in ProcessedStatus]
        )

        self.cursor.executemany(
            'INSERT INTO DatasetTypes (type) VALUES (?)',
            [(t.value,) for t in DatasetType]
        )

        self.cursor.executemany(
            'INSERT INTO EnvTypes (type) VALUES (?)',
            [(t.value,) for t in EnvType]
        )

        self.cursor.executemany(
            'INSERT INTO ActionTypes (type) VALUES (?)',
            [(t.value,) for t in ActionType]
        )
        self.cursor.executemany(
            'INSERT INTO RewardTypes (type) VALUES (?)',
            [(t.value,) for t in RewardType]
        )
        self.cursor.executemany(
            'INSERT INTO RewardTypes (type) VALUES (?)',
            [(t.value,) for t in RewardFunction]
        )

    def parse_directories(self, directories: list[Path]):
        combined_datasets = []
        self._parse_directories_recursive(directories, combined_datasets)
                    
        # add combined datasets at the end, because they might depend on datasets that are not in the database yet
        for path in combined_datasets:
            self.add_combined_dataset(path, path.stem)
        self.commit()
        
    def _parse_directories_recursive(self, directories: list[Path], combined_datasets: list[Path]):
        for dirname in directories:
            if not dirname.is_dir():
                print(f'Warning: {dirname} is not a directory')
                continue

            for path in dirname.iterdir():
                if path.is_file() and path.suffix == '.hdf5':
                    print(f'Parsing {path}')
                    try:
                        metadata, combined = self.get_dataset_metadata(path)
                    except KeyError as e:
                        print(f'Raised error {e}')
                        print(f'Warning: {path} does not contain correct metadata format. Skipping...')
                        continue
                    except PermissionError:
                        print(f'Warning: {path} is not readable. Skipping...')
                        continue
                    if metadata is None:
                        continue
                    try:
                        self.insert_dataset_metadata(path, path.stem, metadata)
                    except ValueError:
                        continue
                    if combined:
                        combined_datasets.append(path)
                elif path.is_dir():
                    self._parse_directories_recursive([path], combined_datasets)

    def get_dataset_metadata(self, dataset_path: Path) -> tuple[dict[str, Any], bool] | tuple[None, bool]:
        with h5py.File(dataset_path, 'r+') as datafile:
            data = datafile['data']
            attrs = data.attrs
        
            combined = 'combined' in attrs and attrs['combined']
    
            return self._get_dataset_metadata(datafile, dataset_path), combined
        
    def _get_list(self, value: list | str | Any) -> list[Any]:
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            try:
                value = json.loads(value)
                return value
            except json.JSONDecodeError:
                return [value]
        return [value]
        
    def _get_dataset_metadata(self, datafile: h5py.File, dataset_path: Path) -> dict[str, Any] | None:
        data = datafile['data']
        attrs = data.attrs

        if 'is_test' in attrs and all(self._get_list(attrs['is_test'])):
            return None
                
        if 'dataset_id' in attrs:
            dataset_id: str = attrs['dataset_id']
        else:
            # 'mask' always exists for robomimic data. we use this to differentiate b/w 
            # robomimic and any random h5py file
            if 'mask' in datafile:
                self._populate_robomimic_data(datafile, dataset_path)
                dataset_id: str = attrs['dataset_id']
            else:
                raise ValueError('Invalid metadata, env_type not specified & could not infer robomimic.')

        env_name = self._get_list(attrs['env_name'])

        env_type = [EnvType(t) for t in self._get_list(attrs['env_type'])] 

        dataset_type = [DatasetType(t) for t in self._get_list(attrs['dataset_type'])]
        
        size = len(data.keys())

        for ep_name in data.keys():
            episode = data[ep_name]
            # Processed = has observations & reward, ready to go
            # Pre-processed = raw robomimic or robosuite teleop data, after running 
            #   conversion script but before dataset_states_to_obs
            # Raw teleop = only relevant for robosuite teleop data, before conversion
            if 'obs' in episode:
                processed_status = ProcessedStatus.PROCESSED

                obs_keys = [key for key in episode['obs'].keys() if key not in ['object']]

            elif 'mask' not in datafile:
                processed_status = ProcessedStatus.RAW_TELEOP
                obs_keys = []
            else:
                processed_status = ProcessedStatus.PRE_PROCESSED
                obs_keys = []

            break

        if processed_status == ProcessedStatus.PROCESSED:
            reward_type = []
            for t in self._get_list(attrs['reward_type']):
                try:
                    reward_type.append(RewardType(t))
                except ValueError:
                    reward_type.append(RewardFunction(t))

            if 'derived_from' in attrs:
                derived_from = json.loads(attrs['derived_from'])
                if isinstance(derived_from[0], list):
                    # flatten
                    derived_from = [item for sublist in derived_from for item in sublist]
            else:
                derived_from = []
        else:
            reward_type = [RewardType.NO_REWARD]
            derived_from = []

        action_type = [ActionType(t) for t in self._get_list(attrs['action_type'])]

        is_real = self._get_list(attrs['is_real'])
        
        if 'description' in attrs:
            description = attrs['description']
        else:
            description = ''
            
        datafile.close()

        return {
            'dataset_id': dataset_id,
            'env_name': env_name,
            'env_type': env_type,
            'dataset_type': dataset_type,
            'obs_keys': obs_keys,
            'processed_status': processed_status,
            'reward_type': reward_type,
            'action_type': action_type,
            'is_real': is_real,
            'size': size,
            'derived_from': derived_from,
            'description': description,
        }
    
    def _populate_robomimic_data(self, datafile: h5py.File, dataset_path: Path):
        """
        Robomimic downloaded datasets don't have proper metadata. Populate with default inferred metadata.
        """

        data = datafile['data']
        attrs = data.attrs
        env_args = json.loads(attrs['env_args'])

        attrs['dataset_id'] = str(uuid.uuid4())

        attrs['env_name'] = env_args['env_name']

        attrs['env_type'] = EnvType.ROBOMIMIC.value
        attrs['dataset_type'] = DatasetType(dataset_path.parent.stem).value
        
        attrs['size'] = len(data.keys())

        for ep_name in data.keys():
            episode = data[ep_name]
            if 'obs' in episode:
                processed_status = ProcessedStatus.PROCESSED
            else:
                processed_status = ProcessedStatus.PRE_PROCESSED
            break

        if processed_status == ProcessedStatus.PROCESSED:
            shaped_rew = env_args['env_kwargs']['reward_shaping']
            if shaped_rew:
                reward_type = RewardType.DENSE
            else:
                reward_type = RewardType.SPARSE
        else:
            reward_type = RewardType.NO_REWARD
            
        attrs['reward_type'] = reward_type.value
        # assume action type is relative
        attrs['action_type'] = ActionType.RELATIVE.value
        attrs['is_real'] = False
        attrs['description'] = 'downloaded robomimic dataset'
        
        attrs['creator'] = 'robomimic'

    def insert_dataset_metadata(self, dataset_path: Path, name: str, metadata: dict[str, Any]):
        dataset_id = metadata['dataset_id']

        if self.fetch_id('Datasets', 'id', dataset_id, none_ok=True) is not None:
            # duplicates should not happen under proper use but just in case (i.e. copying a hdf5 file)
            print(f"Warning: Dataset with id {dataset_id} already exists in the database")
            raise ValueError(f'Duplicate ids')
         
        self.insert_data('Datasets', {
            'id': dataset_id,
            'name': name,
            'path': str(dataset_path),
            'env_name': metadata['env_name'][0],
            'is_real': metadata['is_real'][0],
            'size': metadata['size'],
            'description': metadata['description'],
        })

        category_id = self.fetch_id('DatasetProcessedStatuses', 'category', metadata['processed_status'].value)

        self.insert_data('DatasetProcessedStatusesAssociations', {
            'dataset_id': dataset_id,
            'category_id': category_id,
        })

        env_type_ids = set(
            self.fetch_id('EnvTypes', 'type', t.value) for t in metadata['env_type']
        )
        for env_type_id in env_type_ids:
            self.insert_data('DatasetEnvType', {
                'dataset_id': dataset_id,
                'env_type_id': env_type_id
            })

        dataset_type_ids = set(
            self.fetch_id('DatasetTypes', 'type', t.value) for t in metadata['dataset_type']
        )
        for dataset_type_id in dataset_type_ids:
            self.insert_data('DatasetTypeAssociations', {
                'dataset_id': dataset_id,
                'dataset_type_id': dataset_type_id
            })

        action_type_ids = set(
            self.fetch_id('ActionTypes', 'type', t.value) for t in metadata['action_type']
        )
        for action_type_id in action_type_ids:
            self.insert_data('DatasetActionType', {
                'dataset_id': dataset_id,
                'action_type_id': action_type_id
            })

        if metadata['processed_status'] == ProcessedStatus.PROCESSED:
            reward_type_ids = set(
                self.fetch_id('RewardTypes', 'type', t.value) for t in metadata['reward_type']
            )
            for reward_type_id in reward_type_ids:
                self.insert_data('DatasetRewardType', {
                    'dataset_id': dataset_id,
                    'reward_type_id': reward_type_id
                })

            for key in metadata['obs_keys']:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO ObservationKeys (key)
                    VALUES (?)
                    ''', (key,))

                key_id = self.fetch_id('ObservationKeys', 'key', key)

                self.insert_data('DatasetObservationKeys', {
                    'dataset_id': dataset_id,
                    'observation_key_id': key_id
                })
                
            for id in metadata['derived_from']:
                self.cursor.execute('''
                    INSERT INTO DerivedDatasets (dataset_id, derived_from_id)
                    VALUES (?, ?)
                    ''', (dataset_id, id))
            
    def add_combined_dataset(self, dataset_path: str | Path, name: str):
        """
        Add a combined dataset to the database. If any of the constituent datasets are missing, 
        a dummy dataset is created.

        Args:
            dataset_path (str | Path): path to the combined dataset
            name (str): name of the combined dataset
        """
        print(f'Adding combined dataset {name}')
        with h5py.File(dataset_path, 'r') as datafile:
            data = datafile['data']
            attrs = data.attrs
            if 'combined' not in attrs or not attrs['combined']:
                return
                      

            constituent_dataset_ids = json.loads(attrs['constituent_dataset_id'])
            for i, dataset_id in enumerate(constituent_dataset_ids):
                # if the constituent dataset no longer exist, create a dummy dataset with the same metadata
                if self.fetch_id('Datasets', 'id', dataset_id, none_ok=True) is None:
                    print(f"Warning: Dataset with id {dataset_id} not found in the database. Adding dummy")
                    self._add_dummy_dataset(data, dataset_id, i)
                    
            self._insert_combined_dataset(attrs['dataset_id'], name, str(dataset_path), constituent_dataset_ids)   
            
    def _add_dummy_dataset(self, data: h5py.Group, dataset_id: str, i: int):
        """ 
        Called when a combined dataset is missing a constituent. Creates a dummy
        dataset with the same metadata as the missing constituent.
        
        data - data group object for the combined dataset
        dataset_id - id of the missing dataset
        i - index of the constituent dataset in the combined dataset
        """
        attrs = data.attrs
        constituent_attrs = self._get_ith_constituent_attrs(attrs, i)
        env_name = constituent_attrs['env_name']
        
        self.insert_data('Datasets', {
            'id': dataset_id,
            'name': 'dummy',
            'path': 'dummy_path',
            'env_name': env_name,
            'is_real': constituent_attrs['is_real'],
            'size': 0,
            'description': ''
        })
        # only processed datasets can be combined, so the constituent must be processed
        category_id = self.fetch_id('DatasetProcessedStatuses', 'category', ProcessedStatus.PROCESSED.value)
        env_type_id = self.fetch_id('EnvTypes', 'type', constituent_attrs['env_type'])
        self.insert_data('DatasetProcessedStatusesAssociations', {
            'dataset_id': dataset_id,
            'category_id': category_id,
        })
        self.insert_data('DatasetEnvType', {
            'dataset_id': dataset_id,
            'env_type_id': env_type_id
        })
        dataset_type_id = self.fetch_id('DatasetTypes', 'type', constituent_attrs['dataset_type'])
        self.insert_data('DatasetTypeAssociations', {
            'dataset_id': dataset_id,
            'dataset_type_id': dataset_type_id
        })
        action_type_id = self.fetch_id('ActionTypes', 'type', constituent_attrs['action_type'])
        self.insert_data('DatasetActionType', {
            'dataset_id': dataset_id,
            'action_type_id': action_type_id
        })
        reward_type_id = self.fetch_id('RewardTypes', 'type', constituent_attrs['reward_type'])
        self.insert_data('DatasetRewardType', {
            'dataset_id': dataset_id,
            'reward_type_id': reward_type_id
        })
        
        for ep_key in data.keys():
            episode = data[ep_key]
            obs_keys = [key for key in episode['obs'].keys() if key not in ['object']]
            break
        for key in obs_keys:
            self.cursor.execute('''
                INSERT OR IGNORE INTO ObservationKeys (key)
                VALUES (?)
                ''', (key,))
            key_id = self.fetch_id('ObservationKeys', 'key', key)
            self.insert_data('DatasetObservationKeys', {
                'dataset_id': dataset_id,
                'observation_key_id': key_id
            })
            
    def _insert_combined_dataset(self, combined_id: str, name: str, dataset_path: str, constituent_ids: list[str]):     
        # TODO: currently we are assuming combined datasets cannot be derived from other combined datasets
        # otherwise, we would have to insert into the DerivedDatasets table
           
        self.insert_data(
            'CombinedDatasets',
            {
                'id': combined_id,
                'name': name,
                'path': dataset_path
            }
        )

        for dataset_id in constituent_ids:
            self.insert_data(
                'CombinedDatasetComponents',
                {
                    'combined_dataset_id': combined_id,
                    'dataset_id': dataset_id
                }
            )
            
    def _get_ith_constituent_attrs(self, attrs: dict[str, Any], i: int) -> dict[str, Any]:
        """
        Extract metadata for the i-th constituent dataset from the attrs of a combined dataset.
        """
        constituent_metadata = {}
        
        for attr in ['env_name', 'env_args', 'env_type', 'dataset_type', 'action_type', 'reward_type', 'is_real']:
            try:
                constituent_metadata[attr] = json.loads(attrs[attr])[0]
            except:
                constituent_metadata[attr] = json.loads(attrs[attr])
        
        return constituent_metadata
                
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
        
    def query_datasets(self, return_all=False, combined_only: bool | None=None,**kwargs) -> list[tuple[str]]:
        """
        Perform a query on the database and return matches. 
        
        return_all - if True, return ['id', 'name', 'path', 'is_real', 'env_name', 'action_types', 'obs_keys', 'derived_from', 'constituent_ids']
                    else return ['id', 'name', 'path']
        combined_only - if True, only return combined datasets. If False, only return non-combined datasets. If None, return all.
        kwargs - query parameters. For example, to query by dataset id, pass id='<dataset_id>'. Pass a list to query by multiple values,
                where any dataset whose values of that field are a subset of the query, it will match.
        """
        if return_all:
            query_t = namedtuple('query_t', ['id', 'name', 'path', 'is_real', 'env_name', 'action_types', 'obs_keys', 'derived_from', 'constituent_ids'])
            query = """
            SELECT DISTINCT d.id, d.name, d.path, d.is_real, d.env_name,
            GROUP_CONCAT(DISTINCT at.type) as action_types, 
            GROUP_CONCAT(DISTINCT ok.key) as obs_keys,
            GROUP_CONCAT(DISTINCT df.derived_from_id) as derived_from,
            GROUP_CONCAT(DISTINCT cdc.dataset_id) as constituent_ids
            FROM Datasets d
            LEFT JOIN DatasetActionType dat ON d.id = dat.dataset_id
            LEFT JOIN ActionTypes at ON dat.action_type_id = at.id
            LEFT JOIN DatasetObservationKeys dok ON d.id = dok.dataset_id
            LEFT JOIN ObservationKeys ok ON dok.observation_key_id = ok.id
            LEFT JOIN DerivedDatasets df ON d.id = df.dataset_id
            LEFT JOIN CombinedDatasetComponents cdc ON d.id = cdc.combined_dataset_id
            """
        else:
            query_t = namedtuple('query_t', ['id', 'name', 'path'])
            query = """
            SELECT DISTINCT d.id, d.name, d.path
            FROM Datasets d
            """
            
        if combined_only is not None:
            query += """
            LEFT JOIN CombinedDatasetComponents cdc2 ON d.id = cdc2.combined_dataset_id
            """
        query += "WHERE 1=1"

        params = []

        def add_simple_condition(field, table, column):
            nonlocal query, params
            if field in kwargs:
                values = kwargs[field] if isinstance(kwargs[field], list) else [kwargs[field]]
                placeholders = ','.join(['?' for _ in values])
                query += f" AND d.id IN (SELECT id FROM {table} WHERE {column} IN ({placeholders}))"
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
                
        def add_exact_match_condition(field, table, column):
            """
            the query is a subset of the values
            """
            nonlocal query, params
            if field in kwargs:
                values = kwargs[field] if isinstance(kwargs[field], list) else [kwargs[field]]
                placeholders = ','.join(['?' for _ in values])
                query += f"""
                    AND (
                        SELECT COUNT(DISTINCT {column})
                        FROM {table}
                        WHERE dataset_id = d.id
                        AND {column} IN ({placeholders})
                    ) = {len(values)}
                """
                params.extend(values)
                
        add_simple_condition('id', 'Datasets', 'id')
        add_simple_condition('name', 'Datasets', 'name')
        add_subset_condition('env_type', 'DatasetEnvType dm JOIN EnvTypes et ON dm.env_type_id = et.id', 'et.type', 'et.id')
        add_subset_condition('action_type', 'DatasetActionType dat JOIN ActionTypes at ON dat.action_type_id = at.id', 'at.type', 'at.id')
        add_subset_condition('reward_type', 'DatasetRewardType drt JOIN RewardTypes rt ON drt.reward_type_id = rt.id', 'rt.type', 'rt.id')
        add_subset_condition('dataset_type', 'DatasetTypeAssociations dta JOIN DatasetTypes dt ON dta.dataset_type_id = dt.id', 'dt.type', 'dt.id')
        
        # add_subset_condition('obs_keys', 'DatasetObservationKeys dok JOIN ObservationKeys ok ON dok.observation_key_id = ok.id', 'ok.key', 'ok.id')
        add_exact_match_condition('obs_keys', 'DatasetObservationKeys dok JOIN ObservationKeys ok ON dok.observation_key_id = ok.id', 'ok.key')
        add_subset_condition('processed_status', 'DatasetProcessedStatusesAssociations dpsa JOIN DatasetProcessedStatuses dps ON dpsa.category_id = dps.id', 'dps.category', 'dps.id')

        # exclude any in exclude_obs_keys
        if 'exclude_obs_keys' in kwargs:
            exclude_obs_keys = kwargs['exclude_obs_keys'] if isinstance(kwargs['exclude_obs_keys'], list) else [kwargs['exclude_obs_keys']]
            placeholders = ','.join(['?' for _ in exclude_obs_keys])
            query += f"""
                AND NOT EXISTS (
                    SELECT 1
                    FROM DatasetObservationKeys dok
                    JOIN ObservationKeys ok ON dok.observation_key_id = ok.id
                    WHERE dok.dataset_id = d.id
                    AND ok.key IN ({placeholders})
                )
            """
            params.extend(exclude_obs_keys)

        if 'is_real' in kwargs:
            is_real_values = kwargs['is_real'] if isinstance(kwargs['is_real'], list) else [kwargs['is_real']]
            placeholders = ','.join(['?' for _ in is_real_values])
            query += f" AND d.is_real IN ({placeholders})"
            params.extend([int(val) for val in is_real_values])            
            
        
        if combined_only is not None:
            query += f" AND {'EXISTS' if combined_only else 'NOT EXISTS'} (SELECT 1 FROM CombinedDatasetComponents WHERE combined_dataset_id = d.id)"
            
        query += " GROUP BY d.id, d.name, d.path"

        self.cursor.execute(query, params)
        results = self.cursor.fetchall()

        # name tuple
        results = [query_t(*result) for result in results]
        return results
    
    def _group_matched_datasets(self, matched: list[tuple[str, list[str]]]) -> list[set[str]]:
        """
        Groups together datasets that represent the same underlying data (but might be in different
        forms, like different observation keys or action types)
        
        matched - list of tuples (dataset_id, derived_from) representing datasets that match the query
        
        returns a list of sets of dataset ids, where dataset ids in each set represent the same
        underlying data
        """
        groups: dict[str, set[str]] = {}
        for dataset_id, derived_from in matched:
            derived_from = set(derived_from)
            # Check if this dataset or any of its predecessors are already in a group
            existing_group = None
            for id in {dataset_id} | derived_from:
                if id in groups:
                    existing_group = groups[id]
                    break

            if existing_group is None:
                # if no existing group is found, create a new one
                new_group = {dataset_id} | derived_from
                for id in new_group:
                    groups[id] = new_group
            else:
                # if an existing group is found, merge this dataset and its predecessors into it
                existing_group.add(dataset_id)
                existing_group.update(derived_from)
                for id in existing_group:
                    groups[id] = existing_group
        
        # Return unique groups
        return list(set(frozenset(group) for group in groups.values()))
    
    def _propose_dataset_combinations(
            self, 
            matched: list[tuple[str, list[str]]], 
            combined_datasets: list[tuple[str, list[str]]], 
            is_dummy: dict[str, bool]
        ) -> list[list[str]]:
        """
        Returns a list of possible combinations of datasets that can be combined into a new dataset.
        Avoids proposing combinations that would result in overlapping data. Also avoids dummy datasets.

        matched - list of tuples (dataset_id, derived_from) that match query, non-combined only
        combined_datasets - list of (dataset_id, constituent_ids) that match query, combined only
        is_dummy - dict mapping dataset_id to whether it is a dummy dataset for each dataset in matched
        """
        groups = self._group_matched_datasets(matched)
        
        # For each group, select the dataset(s) that were in the original list
        # This effectively filters out the 'derived_from' ids that might not be in the original list
        matched_ids = [i[0] for i in matched]
        valid_datasets_per_group = [
            set(id for id in group if id in matched_ids)
            for group in groups
        ]

        # remove dummy datasets
        remaining_groups = []
        dummy_only_groups = []
        for group in valid_datasets_per_group:
            group_no_dummy = set()
            for id in group:
                if not is_dummy[id]:
                    group_no_dummy.add(id)
            if len(group_no_dummy) > 0:
                remaining_groups.append(group_no_dummy)
            else:
                dummy_only_groups.append(group)

        def generate_combinations(
                current_groups: list[tuple[str, list[str]]], 
                current_combined: list[tuple[str, list[str]]], 
                excluded_ids: set[str]
            ) -> list[list[str]]:
            """
            Recursively generate all possible dataset combinations.

            current_groups - groups of datasets ids (not combined) that can be combined
            current_combined - combined datasets that can be combined (due to having constituents in dummy-only groups)
            excluded_ids - dataset ids that have already been combined and thus should be excluded
            """
            if not current_groups and not current_combined:
                return [[]]

            result = []
                        
            for cd in current_combined:
                constituent_ids = set(cd[1])
                if not constituent_ids & excluded_ids:
                    # exclude constituents 
                    new_excluded = excluded_ids | constituent_ids
                    new_groups = [g for g in current_groups if not g & constituent_ids]
                    # remove potential combined datasets that share constituents with this one
                    new_combined = [c for c in current_combined if c[0] != cd[0] and not set(c[1]) & constituent_ids]
                    result.extend([[cd[0]] + combo 
                                for combo in generate_combinations(new_groups, new_combined, new_excluded)])

            # try adding from non-dummy groups
            if current_groups and not current_combined:
                group = current_groups[0]
                for dataset_id in group:
                    if dataset_id not in excluded_ids:
                        result.extend([[dataset_id] + combo 
                                    for combo in generate_combinations(current_groups[1:], current_combined, excluded_ids)])
                    
            return result
        
        # Find combined datasets that could replace dummy-only groups
        relevant_combined = [cd for cd in combined_datasets 
                            if any(set(cd[1]) & group for group in dummy_only_groups)]
        # include combined datasets that do not share any constituents with matched
        relevant_combined.extend([cd for cd in combined_datasets if not set(cd[1]) & set(matched_ids)])

        # Generate all possible combinations
        return generate_combinations(remaining_groups, relevant_combined, set())
    
    def _get_valid_combinations(self, combinations: list[list[str]]) -> list[list[str]]:
        """
        Given a list of proposed dataset commbinations, return only the ones that can
        be combined. I.e., no conflicting action types, same environment, etc.
        
        *does not check for overlapping data. this assumes the combos are returned
        from _propose_dataset_combinations, which already should avoid overlapping data
        
        combinations - list of proposed dataset combinations. Each combination is a list
        of datatset_ids
        """
        
        valid_combinations = []
        for combo in combinations:
            datasets_in_combo = self.query_datasets(return_all=True, id=combo)
            
            action_types = [q.action_types for q in datasets_in_combo]
            obs_keys = [q.obs_keys for q in datasets_in_combo]
            env_names = [q.env_name for q in datasets_in_combo]
            is_reals = [q.is_real for q in datasets_in_combo]
            
            to_match: list[list[str]] = [action_types, obs_keys, env_names, is_reals]
            
            for match_condition in to_match:
                # make sure everythin in match_condition is the same
                for item in match_condition:
                    items = set(item.split(','))
                    match_condition_item = items
                    break
                for item in match_condition:
                    items = set(item.split(','))
                    if items ^ match_condition_item:
                        # not all items are the same
                        break
                else:
                    # no break - match_condition passes
                    continue
                
                # break - match_condition failed
                break
            else:
                # no break - all match_conditions passed
                valid_combinations.append(combo)
                
        return valid_combinations
    
    def _check_combined_exists(self, dataset_ids: list[str], combined: list[tuple[str, list[str]]]) -> str | None:
        """
        Check if a combined dataset with the provided constituents already exists.
        If exists, returns the id of the matching combined dataset. Otherwise, returns None.
        
        dataset_ids - list of dataset ids that would be combined
        combined - possible combined dataset ids that could contain the constituents
        """
        for combined_id, constituents in combined:
            if set(constituents) == set(dataset_ids):
                return combined_id
        return None
    
    def get_dataset(self, attempt_combination=True, **kwargs) -> str | None:
        """
        Return the path of the dataset that matches the query parameters. If multiple datasets
        match, try to combine them into a new dataset. If no matching datasets are found,
        attempt to convert existing datasets into a matching new dataset.
        
        If impossible, returns None.
        """
        all_single_matched = self.query_datasets(combined_only=False, return_all=True, processed_status=ProcessedStatus.PROCESSED.value, **kwargs)
        all_combined_matched = self.query_datasets(combined_only=True, return_all=True, processed_status=ProcessedStatus.PROCESSED.value, **kwargs)
        is_dummy = {}
        single_matches = []
        single_id_to_path: dict[str, str] = {}
        single_id_to_name: dict[str, str] = {}
        for match in all_single_matched:
            if match.derived_from is None:
                derived_from = []
            else:
                derived_from = match.derived_from.split(',')
            single_matches.append((match.id, derived_from))
            # TODO: perhaps find a better marker for a dummy than setting name to 'dummy'?
            is_dummy[match.id] = match.name == 'dummy'
            single_id_to_path[match.id] = match.path
            single_id_to_name[match.id] = match.name
        combined_matches = []
        combined_id_to_path = {}
        combined_id_to_name = {}
        for match in all_combined_matched:
            constituent_ids = match.constituent_ids.split(',')
            combined_matches.append((match.id, constituent_ids))
            combined_id_to_path[match.id] = match.path
            combined_id_to_name[match.id] = match.name
        id_to_path = single_id_to_path | combined_id_to_path
        id_to_name = single_id_to_name | combined_id_to_name
                
        proposed_combos = self._propose_dataset_combinations(single_matches, combined_matches, is_dummy)
        valid_combos = self._get_valid_combinations(proposed_combos)
                
        if not valid_combos or all(not item for item in valid_combos):
            return None
            # TODO: attempt to convert 
                        
        if len(valid_combos) == 1 and len(valid_combos[0]) == 1:
            return id_to_path[valid_combos[0][0]]
        
        if not attempt_combination:
            return None
        
        # go thru valid combos, if any already exist, return the first one
        for combo in valid_combos:
            combined_id = self._check_combined_exists(combo, combined_matches)
            if combined_id:
                return combined_id_to_path[combined_id]
            
        # otherwise, try to combine the first one
        # TODO: smarter combo selection, perhaps one that minimizes fractionalization of data attributes
        combo = valid_combos[0]
        dataset_paths = [id_to_path[dataset_id] for dataset_id in combo]
        new_name = '_'.join([id_to_name[dataset_id] for dataset_id in combo]) + '_COMBINED'
        combined_dir = Path(COMBINED_DATA_DIR)
        new_combined_path = combined_dir / f"{new_name}.hdf5"
        print(f'Combining datasets & saving to path: {new_combined_path}')
        if not COMBINED_DATA_DIR.exists():
            COMBINED_DATA_DIR.mkdir()
        combine_datasets(dataset_paths, str(new_combined_path))
        
        # Insert the new combined dataset into the database
        metadata, _ = self.get_dataset_metadata(new_combined_path)
        self.insert_dataset_metadata(new_combined_path, new_name, metadata)
        self.add_combined_dataset(new_combined_path, new_name)
        
        self.commit()
        
        return new_combined_path

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
        
    def visualize_query_result(self, query, params=None, maxcolwidths=25, to_csv=False):
        # Execute the provided query
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # Get column headers from the cursor description
        columns = [column[0] for column in self.cursor.description]

        if to_csv:
            with open('all_datasets.csv', 'w') as f:
                content = tabulate.tabulate(rows, headers=columns, tablefmt='csv')
                with open('all_datasets.csv', 'w') as f:
                    f.write(content)
            print(f"\nQuery result saved to all_datasets.csv")
        else:
            print(f"\nQuery result:")
            print(tabulate.tabulate(rows, headers=columns, tablefmt="grid", maxcolwidths=maxcolwidths))
            print(f"Total rows: {len(rows)}")
        
        
    def visualize_all(self, to_csv=False, show_id=False):        
        id_str = 'd.id, ' if show_id else ''
        
        query = f"""
            SELECT DISTINCT {id_str}d.name, d.path, d.size, d.is_real, d.env_name, dt.type, dps.category, d.description,
            GROUP_CONCAT(DISTINCT at.type) as action_types, 
            GROUP_CONCAT(DISTINCT ok.key) as obs_keys,
            GROUP_CONCAT(DISTINCT rt.type) as reward_types
            FROM Datasets d
            LEFT JOIN DatasetActionType dat ON d.id = dat.dataset_id
            LEFT JOIN ActionTypes at ON dat.action_type_id = at.id
            LEFT JOIN DatasetObservationKeys dok ON d.id = dok.dataset_id
            LEFT JOIN ObservationKeys ok ON dok.observation_key_id = ok.id
            LEFT JOIN DatasetProcessedStatusesAssociations dpsa ON dpsa.dataset_id = d.id
            LEFT JOIN DatasetProcessedStatuses dps ON dpsa.category_id = dps.id
            LEFT JOIN DatasetRewardType drt ON d.id = drt.dataset_id
            LEFT JOIN RewardTypes rt ON drt.reward_type_id = rt.id
            LEFT JOIN DatasetTypeAssociations dta ON dta.dataset_id = d.id
            LEFT JOIN DatasetTypes dt ON dta.dataset_type_id = dt.id
            GROUP BY d.id, d.name, d.path
        """
        maxcolwidths = [15, 35, 8, 10, 20, 8, 30, 15, 10, 20]
        maxcolwidths = [50] + maxcolwidths if show_id else maxcolwidths
        self.visualize_query_result(query, maxcolwidths=maxcolwidths, to_csv=to_csv)
        
    def check_updated(self) -> bool:
        """
        Check if all datasets in the database are still present in their original paths.
        
        If not, should re-parse the directories to update the database.
        
        Note, does not check if newly added datasets exist.
        """
        all_datasets = self.query_datasets()
        for dataset in all_datasets:
            if dataset.name == 'dummy':
                continue
            path = Path(dataset.path)
            if not path.exists():
                return False
        return True
    
    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.close()
        
###
# fix bug in tabulate by monkey patching
###
def _wrap_text_to_colwidths(list_of_lists, colwidths, numparses=True):
    numparses = tabulate._expand_iterable(numparses, len(list_of_lists[0]), True)

    result = []

    for row in list_of_lists:
        new_row = []
        for cell, width, numparse in zip(row, colwidths, numparses):
            if tabulate._isnumber(cell) and numparse:
                new_row.append(cell)
                continue

            if width is not None:
                wrapper = tabulate._CustomTextWrap(width=width)
                # Cast based on our internal type handling
                # Any future custom formatting of types (such as datetimes)
                # may need to be more explicit than just `str` of the object
                # casted_cell = (
                #     str(cell) if _isnumber(cell) else _type(cell, numparse)(cell)
                # )
                if cell is not None:
                    wrapped = wrapper.wrap(cell)
                new_row.append("\n".join(wrapped))
            else:
                new_row.append(cell)
        result.append(new_row)

    return result
tabulate._wrap_text_to_colwidths = _wrap_text_to_colwidths
