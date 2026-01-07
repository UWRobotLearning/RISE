PRAGMA foreign_keys = ON;

CREATE TABLE Datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    env_name TEXT NOT NULL,
    path TEXT NOT NULL,
    is_real BOOLEAN NOT NULL,
    size TEXT NOT NULL,
    description TEXT NOT NULL
);

--- Enumerate types

CREATE TABLE DatasetProcessedStatuses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT UNIQUE NOT NULL
);

CREATE TABLE DatasetTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE NOT NULL
);

CREATE TABLE RewardTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE NOT NULL
);

CREATE TABLE ObservationKeys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL
);

CREATE TABLE ActionTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE NOT NULL
);

CREATE TABLE EnvTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE NOT NULL
);

--- Metadata assocations

CREATE TABLE DatasetProcessedStatusesAssociations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES DatasetProcessedStatuses(id)
);

CREATE TABLE DatasetEnvType (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    env_type_id INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (env_type_id) REFERENCES EnvTypes(id)
);

CREATE TABLE DatasetTypeAssociations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    dataset_type_id INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_type_id) REFERENCES DatasetTypes(id)
);

CREATE TABLE DatasetObservationKeys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    observation_key_id INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (observation_key_id) REFERENCES ObservationKeys(id)
);

CREATE TABLE DatasetActionType (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    action_type_id INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (action_type_id) REFERENCES ActionTypes(id)
);

CREATE TABLE DatasetRewardType (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL,
    reward_type_id INTEGER,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (reward_type_id) REFERENCES RewardTypes(id)
);

CREATE TABLE DerivedDatasets (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    derived_from_id TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE
);

CREATE TABLE CombinedDatasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE CombinedDatasetComponents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    combined_dataset_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    FOREIGN KEY (combined_dataset_id) REFERENCES CombinedDatasets(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES Datasets(id) ON DELETE CASCADE
);