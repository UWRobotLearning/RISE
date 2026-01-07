import os
import re
from pathlib import Path

def get_latest_sb_policy(logdir: Path, find_epoch=False) -> str | int:
    """
    Get the latest sb3 policy checkpoint from a logdir.
    """
    if not logdir.exists():
        raise ValueError(f'The path {logdir} does not exist')
    policy_files = list(logdir.glob('*.zip'))
    if 'policy.zip' in [file.name for file in  policy_files]:
        latest = logdir / 'policy.zip'
    else:
        latest = None
    if not find_epoch and latest is not None:
        return latest

    pattern = re.compile('(\d+).zip')
    
    largest_epoch = -1
    largest_policy = None
    for filename in policy_files:
        match = pattern.match(filename.name)
        if match is not None:
            epoch = int(match.groups()[0])
            if epoch > largest_epoch:
                largest_epoch = epoch
                largest_policy = filename
    
    if largest_policy is None:
        raise ValueError(f'No policy found in {logdir}')
    if find_epoch:
        return largest_epoch
    if latest is not None:
        return latest
    return largest_policy

def get_latest_robomimic_policy(logdir: Path, find_epoch=False) -> str | int:
    """
    Get the latest robomimic policy checkpoint from a logdir.
    """
    if not logdir.exists():
        raise ValueError(f'The path {logdir} does not exist')
    policy_files = list(logdir.glob('*.pth'))
    pattern = re.compile('.*model_epoch_(\d+).*\.pth')
    
    largest_epoch = -1
    largest_policy = None
    for filename in policy_files:
        match = pattern.match(filename.name)
        if match is not None:
            epoch = int(match.groups()[0])
            if epoch > largest_epoch:
                largest_epoch = epoch
                largest_policy = filename
    
    if largest_policy is None:
        raise ValueError(f'No policy found in {logdir}')
    if find_epoch:
        return largest_epoch
    return largest_policy

def get_latest_diffusion_policy(logdir: Path, find_epoch=False) -> str | int:
    """
    Get the latest diffusion policy checkpoint from a logdir.
    """
    if not logdir.exists():
        raise ValueError(f'The path {logdir} does not exist')
    policy_files = list(logdir.glob('*.ckpt'))
    if 'latest.ckpt' in [file.name for file in  policy_files]:
        latest = logdir / 'latest.ckpt'
    else:
        latest = None
    if not find_epoch and latest is not None:
        return latest
    pattern = re.compile('.*epoch=(\d+).*\.ckpt')
    
    largest_epoch = -1
    largest_policy = None
    for filename in policy_files:
        match = pattern.match(filename.name)
        if match is not None:
            epoch = int(match.groups()[0])
            if epoch > largest_epoch:
                largest_epoch = epoch
                largest_policy = filename
    
    if largest_policy is None:
        raise ValueError(f'No policy found in {logdir}')
    if find_epoch:
        return largest_epoch
    if latest is not None:
        return latest
    return largest_policy

def get_latest_q_transformer_policy(logdir: Path, find_epoch=False) -> str | int:
    """
    Get the latest qtransformer policy checkpoint from a logdir.
    """
    if not logdir.exists():
        raise ValueError(f'The path {logdir} does not exist')
    policy_files = list(logdir.glob('*.pt'))
    pattern = re.compile('.*checkpoint-(\d+).*\.pt')
    
    largest_epoch = -1
    largest_policy = None
    for filename in policy_files:
        match = pattern.match(filename.name)
        if match is not None:
            epoch = int(match.groups()[0])
            if epoch > largest_epoch:
                largest_epoch = epoch
                largest_policy = filename
    
    if largest_policy is None:
        raise ValueError(f'No policy found in {logdir}')
    if find_epoch:
        return largest_epoch
    return largest_policy
    
def get_latest_ogbench_policy(logdir: Path, find_epoch=False) -> str | int:
    """
    Get the latest ogbench policy checkpoint from a logdir.
    """
    if not logdir.exists():
        raise ValueError(f'The path {logdir} does not exist')
    policy_files = list(logdir.glob('*.pkl'))
    pattern = re.compile('.*params_(\d+).*\.pkl')
    
    largest_epoch = -1
    largest_policy = None
    for filename in policy_files:
        match = pattern.match(filename.name)
        if match is not None:
            epoch = int(match.groups()[0])
            if epoch > largest_epoch:
                largest_epoch = epoch
                largest_policy = filename
    
    if largest_policy is None:
        raise ValueError(f'No policy found in {logdir}')
    if find_epoch:
        return largest_epoch
    return largest_policy