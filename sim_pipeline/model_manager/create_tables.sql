PRAGMA foreign_keys = ON;

CREATE TABLE Models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    identifier TEXT NOT NULL,
    creator TEXT NOT NULL,
    path TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE PolicyTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT UNIQUE NOT NULL
);

CREATE TABLE ModelPolicyTypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    policy_type_id INTEGER NOT NULL,
    FOREIGN KEY (model_id) REFERENCES Models(id) ON DELETE CASCADE,
    FOREIGN KEY (policy_type_id) REFERENCES PolicyTypes(id)
);

CREATE TABLE ModelDatasetAssociations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES Models(id) ON DELETE CASCADE
);