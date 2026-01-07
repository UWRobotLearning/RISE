### Installation

Clone repo

```
```

Install Dependencies

```
cd ood-sim-pipeline

# Install base dependencies
conda env create -f environment.yaml
conda activate 

# pytorch3d (dependency for diffusion) must be installed separately 
# due to bugs with downgrading torch to cpu
conda install pytorch3d -c pytorch3d

# install sim-pipeline package
pip install -e .

cd ..

# install custom robosuite from source
git clone git@github.com:UWRobotLearning/robosuite.git
cd robosuite
pip install -e .

# install custom robomimic from source
git clone git@github.com:UWRobotLearning/robomimic.git
cd robomimic
pip install -e .

# install mimicgen_environments
git clone git@github.com:UWRobotLearning/mimicgen_environments.git
cd mimicgen_environments
pip install -e .
```

### Datasets / Data Collection

Datasets used in paper can be downloaded here (pending).

Simulation data in robosuite can be collected using Spacemouse in the same manner as in robosuite. With the proper spacemouse vendor ID and product ID set in `robosuite/utils/macros.py`, run:

```
python scripts/collect_human_demonstrations.py +env=<env_config>

# Then convert to the proper format
python sim_pipeline/data_manager/convert_robosuite.py --dataset /path/to/collected/data

python scripts/dataset_states_to_obs.py --dataset /path/to/data --output_name <name> --camera_names agentview,robot0_eye_in_hand
```

Envs can be found in `configs/envs`.


### Training and evaluation

Configs for each experiment can be found in `configs/training/_imitation_learning/_offline_rl/_experiments`. For custom datasets, replace `low_dim_keys` with the names of observation proprioception keys, and `rgb_keys` with the names of observation image keys in the config. To train, run

```
python scripts/train_robomimic.py -cn iql_diffusion +training=<square/piece_assembly/threading/real_franka/...> data.dataset_path=<path_to_dataset>
```

To evaluate policies in simulation, run:

```
python scripts/eval.py -cn eval_robomimic_iql_diffusion exp.model_path=<path_to_checkpoint> +env=<env_config_to_eval_in>
```

valid envs can be found in `configs/env`. Policies used in the paper can be downloaded here (pending).

