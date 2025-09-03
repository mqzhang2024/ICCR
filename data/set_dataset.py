import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,root,path):
        self.data = []
        self.root=root
        with open('./data/'+root+'/'+path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename =item['source']
        target_filename = item['target']

        # Assembling biological information into standardised text
        sex= 'male' if item['gender']=='M' else 'female'
        age=item['age']
        if age<14:
            age_level=1
        elif age<30:
            age_level=2
        elif age<50:
            age_level=3
        elif age<65:
            age_level=4
        else:
            age_level=5
       
        prompt="3D face scan,Asian "+sex+",age level "+str(age_level)+","

        if item['body_mass']=='1':
            prompt+= 'very thin face.'
        elif item['body_mass']=='2':
            prompt+= 'thin face.' 
        elif item['body_mass']=='3':
            prompt+= 'slightly thin face.' 
        elif item['body_mass']=='4':
            prompt+= 'normal face.'
        elif item['body_mass']=='5':
            prompt+= 'slightly fat face.' 

        prob=torch.zeros(1).float().uniform_(0, 1) < 0.5
        prompt="" if prob else prompt
        
        # Skulls and Craniofacial Images
        source = cv2.imread('./data/pairedCS/skulls/'+source_filename)
        target = cv2.imread('./data/pairedCS/craniofacial/'+target_filename)
    
        # OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
       
        # Normalize images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return dict(jpg=target, txt=prompt, hint=source)
    