import hydra

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from sim_pipeline.configs._q_transformer.q_transformer import QTransformerBaseConfig
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.utils.shape_meta import get_shape_meta, update_diffusion_cfg_with_shape_meta
from sim_pipeline.utils.diffusion_utils import get_policy_class, get_workspace
from sim_pipeline.utils.train_utils import get_latest_diffusion_policy, get_latest_q_transformer_policy
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained

from robomimic.utils.dataset import SequenceDataset
from q_transformer import QLearner, QRoboticTransformer

@hydra.main(version_base=None, config_path='../configs', config_name='q_transformer')
def train(config: QTransformerBaseConfig):
    import traceback
    import sys
    try:
        run(config)
    except:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: QTransformerBaseConfig):
    # device = setup_experiment(cfg.exp)

    dataset_path = get_dataset(cfg.data)
    
    model = QRoboticTransformer(
        vit = dict(
            num_classes = 1000,
            dim_conv_stem = 64,
            dim = 64,
            dim_head = 64,
            depth = (2, 2, 5, 2),
            window_size = 6,
            mbconv_expansion_rate = 4,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.1
        ),
        num_actions = 8,
        depth = 1,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
        dueling = True,
        weight_tie_action_bin_embed = False,
        condition_on_text=False,
        num_residual_streams = 1
    )
    lowdim_keys = cfg.training.low_dim_keys
    rgb_keys = cfg.training.rgb_keys
    
    if lowdim_keys:
        raise NotImplementedError('Lowdim keys not implemented')
    if not rgb_keys:
        raise NotImplementedError('RGB keys needed')
    
    torch_dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=rgb_keys,
        dataset_keys=['actions', 'rewards', 'dones'],
        load_next_obs=True,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode='all',
        q_transformer=True,
    )
        
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    log_dir = Path(cfg.exp.logdir) / cfg.name / dt_string
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        
    q_learner = QLearner(
        model=model,
        dataset=torch_dataset,
        batch_size=cfg.training.batch_size,
        num_train_steps=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        checkpoint_folder=log_dir / 'checkpoints',
        checkpoint_every=500,
    )
        
    create_model_metadata(
        name=cfg.name,
        policy_type=PolicyType.QTRANSFORMER,
        policy_identifier=dt_string,
        dataset_path=dataset_path,
        hydra_config=cfg,
        filepath=log_dir / 'metadata.json',
    )

    try:
        q_learner()
    finally:
        max_epoch_saved: int = get_latest_q_transformer_policy(log_dir / 'checkpoints', find_epoch=True)
        # only consider a training run valid if we have ran for more than 1000 (hardcoded) epochs
        if max_epoch_saved >= 15:
            update_metadata_trained(log_dir / 'metadata.json')  
