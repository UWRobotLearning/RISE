import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class PointMazeDataset(Dataset):
    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'r')
        self.observations = self.file['observations']
        self.next_observations = self.file['next_observations']
        self.achieved_goals = self.file['achieved_goals']
        self.desired_goals = self.file['desired_goals']
        self.images = self.file['images']
        self.actions = self.file['actions']
        self.terminals = self.file['terminals']
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observation = torch.tensor(self.observations[idx], dtype=torch.float32)
        next_observation = torch.tensor(self.next_observations[idx], dtype=torch.float32)
        achieved_goal = torch.tensor(self.achieved_goals[idx], dtype=torch.float32)
        desired_goal = torch.tensor(self.desired_goals[idx], dtype=torch.float32)
        image = cv2.resize(self.images[idx], (160, 160), interpolation=cv2.INTER_NEAREST)
        image = image.transpose(2,1,0)
        image = torch.tensor(image, dtype=torch.uint8)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        terminal = torch.tensor(self.terminals[idx], dtype=torch.long)

        obs = torch.cat((observation, desired_goal), dim=0)
        dict = {'observation': observation, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal, 'obs': obs, 'image': image, 'action': action, 'next_observation': next_observation, 'terminal': terminal}
        return dict

# Example usage
if __name__ == "__main__":
    maze_id = 'train_easy'
    fname = 'experiments/' + maze_id + '_dataset.h5'
    dataset = PointMazeDataset(fname)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    dict = dataset[0]
    observation, achieved_goal, desired_goal, image, action, obs = dict['observation'], dict['achieved_goal'], dict['desired_goal'], dict['image'], dict['action'], dict['obs']
    image = image.permute(2,1,0)
    height, width, _ = image.shape
    print(image.shape)
    print(f"Observation: {observation.shape}, Achieved Goal: {achieved_goal.shape}, Desired Goal: {desired_goal.shape}, Image: {image.shape}, Action: {action.shape}")

    video_name = 'experiments/' + maze_id + '_dataset_video.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    for i in range(len(dataset)):
        obs = dataset[i]
        observation, achieved_goal, desired_goal, image, action = obs['observation'], obs['achieved_goal'], obs['desired_goal'], obs['image'], obs['action']
        image = image.permute(2,1,0)
        image = image.detach().numpy()
        image = image[...,::-1]
        video.write(image)

    print(f'Saved video to {video_name}')
    video.release()
    cv2.destroyAllWindows()