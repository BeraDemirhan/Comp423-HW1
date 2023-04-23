import os
import cv2
import json
from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, task='imitation_learning'):
        self.data_root = data_root
        self.samples = []
        for filename in os.listdir(data_root):
            if filename.endswith('.jpg'):
                json_file = filename[:-4] + '.json'
                json_path = os.path.join(data_root, json_file)
                if os.path.exists(json_path):
                    self.samples.append((filename, json_path))
        self.task = task
    
    def __len__(self):
        return len(self.samples)
        

    def __getitem__(self, index):
        """Return RGB images and task-specific data"""
        # Get RGB image
        rgb_path = os.path.join(self.data_root, self.samples[index][0])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if self.task == 'imitation_learning':
            # Get expert actions for imitation learning
            json_path = os.path.join(self.data_root, self.samples[index][1])
            with open(json_path) as f:
                expert_actions = json.load(f)

            return rgb, expert_actions

        elif self.task == 'affordance_prediction':
            # Get driving affordances for affordance prediction
            json_path = os.path.join(self.data_root, self.samples[index][1])
            with open(json_path) as f:
                affordances = json.load(f)

            return rgb, affordances

        else:
            raise ValueError(f"Unknown task '{self.task}'")