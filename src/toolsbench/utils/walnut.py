import torch
import tensordict

import pandas as pd
import random
from pathlib import Path
import numpy as np

from collections import OrderedDict
from itertools import cycle

from tqdm import tqdm
               
class WalnutDatasetMultiP(torch.utils.data.Dataset):
    def __init__(self, 
                 input_dir: str,
                 input_files: list[str],
                 train: bool=True,
                 test: bool=False,
                 batch_size: int=1,
                 num_projs: list[int] = [30,50,100],
                 patch_size: tuple[int,int,int] | int=128,
                 patch_training: bool=True,
                 center_crop: bool=True,
                 output: list[str] = ['input', 'target'],
                 dataset_size: int=400,
                 input_clamp: bool=True,
                 **kwargs):
        super(WalnutDatasetMultiP, self).__init__()
        
        self.input_dir = Path(input_dir)
        self.input_files = {}
        self.dfs = {}
        self.sparse_indexes = {}
        self.batch_size = batch_size
        self._num_projs = cycle(num_projs)

        assert len(input_files) == len(num_projs)
        for num_proj, input_file in zip(num_projs, input_files):
            self.input_files[num_proj] = self.input_dir / input_file
         
            df = pd.read_csv(self.input_files[num_proj])

            if train:
                df = df.loc[df.split_set == "train"]
            elif not train and not test:
                df = df.loc[df.split_set == "validation"]
                df = df.loc[df.id == 12] # FIXME: hardcode sample id to deal with variable geometry later
            else:
                df = df.loc[df.split_set == 'test']
                
            self.dfs[num_proj] = df   
            self.sparse_indexes[num_proj] = torch.linspace(0, 1201, steps=num_proj+1, dtype=torch.long)[:-1]
            
        self.train = train
        self.test = test
        if type(patch_size) is int:
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.patch_training = patch_training
        self.center_crop = center_crop
        self.input_clamp = input_clamp
        
        self.output = output
        self.dataset_size = dataset_size
        self.create_memmap()

    def create_memmap(self):
        self.sparse_rcs = dict()
        self.reference_rcs = dict()
        self.sinogram = dict()
        
        print("Initializing memory-mapping for each volume....", end='\n')
        for num_proj, df in self.dfs.items():
            self.sparse_rcs[num_proj] = dict()
            self.reference_rcs[num_proj] = dict()
            self.sinogram[num_proj] = dict()
        
            sparse_indexes = self.sparse_indexes[num_proj]
        
            for row in tqdm(df.itertuples(), total=len(df)):
                sample_id = row.id

                self.reference_rcs[num_proj][sample_id] \
                = tensordict.MemoryMappedTensor.from_filename(
                    self.input_dir / row.reconstruction_file,
                    shape=(row.number_of_slice, row.num_voxels, row.num_voxels),
                    dtype=torch.float32,
                )               
             
                self.sparse_rcs[num_proj][sample_id] \
                = tensordict.MemoryMappedTensor.from_filename(
                    self.input_dir /  row.sparse_reconstruction_file,
                    shape=(row.number_of_slice, row.num_voxels, row.num_voxels),
                    dtype=torch.float32,
                )
               
                if 'sinogram' in self.output:
                    self.sinogram[num_proj][sample_id] \
                    = torch.from_numpy(np.memmap(self.input_dir / row.sinogram_file,
                                dtype='float32',
                                mode='c',
                                shape=(1200,972,768)))
                    # astra requires (H,P,W) format
                    self.sinogram[num_proj][sample_id] = self.sinogram[num_proj][sample_id][sparse_indexes].permute(1,0,2).contiguous()
                        
    def __getitem__(self, index):
        num_proj = next(self._num_projs)
        
        elements = []
        for batch_idx in range(self.batch_size):
            elements.append(self._getitem(num_proj))
            
        return torch.utils.data.default_collate(elements)
            
    def _getitem(self, num_proj):
        sample_id = random.choice(list(self.reference_rcs[num_proj].keys()))
        data = OrderedDict()

        input = self.sparse_rcs[num_proj][sample_id]
        target = self.reference_rcs[num_proj][sample_id]

        if self.patch_training:
            if len(self.patch_size) == 3:
                D_patch, H_patch, W_patch = self.patch_size
            elif len(self.patch_size) == 2:
                H_patch, W_patch = self.patch_size
                D_patch = 1
            
            z = random.randint(80, 501 - 80 - D_patch)
            if self.center_crop:
                if H_patch == 501:
                    y = 0
                else:
                    y = random.randint(50, 501 - 50 - H_patch)
                if W_patch == 501:
                    x = 0
                else:
                    x = random.randint(50, 501 - 50 - W_patch)
            else:
                y = random.randint(0, 501 - H_patch)
                x = random.randint(0, 501 - W_patch)
            
            # add channel dimension with [None,]
            if len(self.patch_size) == 3:
                input_patch = input[None, z:z+D_patch, y:y+H_patch, x:x+W_patch]
                target_patch = target[None, z:z+D_patch, y:y+H_patch, x:x+W_patch]
            elif len(self.patch_size) == 2:
                input_patch = input[None, z, y:y+H_patch, x:x+W_patch]
                target_patch = target[None, z, y:y+H_patch, x:x+W_patch]

            if 'input' in self.output:
                data['input'] = input_patch

            if 'target' in self.output:
                data['target'] = target_patch
                
            if 'sinogram' in self.output:
                data['sinogram'] = self.sinogram[num_proj][sample_id][None]
                
            if 'patch_indices' in self.output:
                data['patch_indices'] = torch.tensor([z,y,x]).long()
                
            if 'full_target' in self.output:
                data['full_target'] = target[None]

        else:
            if 'input' in self.output:
                data['input'] = input[None]
            if 'target' in self.output:
                data['target'] = target[None]
                
            if 'sinogram' in self.output:
                data['sinogram'] = self.sinogram[num_proj][sample_id][None]

        if self.input_clamp and 'input' in data:
            data['input'] = data['input'].clamp(0,None)

        data['sample_id'] = sample_id
                
        return data
    
    def __len__(self):
        return self.dataset_size
        # native_size = int(501**3)
        # patch_size = np.prod(self.patch_size)
        # return len(self.df) * int(native_size/patch_size)
