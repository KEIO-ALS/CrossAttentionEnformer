from typing import Any
import torch
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
from config import get_config

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

import tensorflow as tf
from tqdm import tqdm

def _TF2Torch(tf_tensor):
   np_tensor = tf_tensor.numpy()
   return torch.from_numpy(np_tensor)

def _decode_fn(record_bytes, organism):
  metadata = get_config("data", "metadata_"+organism)
  example = tf.io.parse_single_example(
      serialized = record_bytes,
      features = {
          'sequence': tf.io.FixedLenFeature([], tf.string),
          'target': tf.io.FixedLenFeature([], tf.string),
      }
  )
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)
  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)
  return (sequence, target)

# def _organism_path(organism):
#     return os.path.join('Basenji')

def _tfrecord_files(organism, subset):
    return sorted(glob.glob(os.path.join(
        "data", "Basenji", f'data_{organism}_tfrecords_{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

class Basenji2Datasets():
  def __init__(self, organism, subset):
    super().__init__()
    self.organism = organism
    self.tfr_files = _tfrecord_files(organism, subset)
  
  def __len__(self):
    return len(self.tfr_files)

  def __getitem__(self, index):
    glob = self.tfr_files[index]
    return self._get_dataset(glob)
  
  def _get_dataset(self, glob):
    # glob = tf.io.gfile.glob(os.path.join(tfr_file))
    return tf.data.TFRecordDataset(
        glob,
        compression_type = "ZLIB",
        num_parallel_reads = get_config("general", "num_workers"),
    ).map(lambda x: _decode_fn(x, self.organism))
  

class Basenji2Dataset():
    def __init__(self, subset) -> None:
        self.subset = subset
        human_dataloader_list = Basenji2Datasets("human", subset)
        mouse_dataloader_list = Basenji2Datasets("mouse", subset)

        self.iters = {
           "human": self._get_iter(human_dataloader_list),
           "mouse": self._get_iter(mouse_dataloader_list),
        }
        self.batch_size = get_config("general","batch_size")

    def _get_iter(self, dataloader_list):
        for dataloader in dataloader_list:
           for data in dataloader:
                yield data

    def _get_count(self, organism):
       return get_config("data","metadata_"+organism,self.subset+"_seqs")
    
    def _get_shorter(self):
        if self._get_count("human") > self._get_count("mouse"):
            return "mouse", "human"
        return "human", "mouse"

    def __len__(self):
       return self._get_count("human")//self.batch_size + self._get_count("mouse")//self.batch_size

    def __getitem__(self, index):
        def get_data(organism):
            x, y = next(self.iters[organism])
            x, y = _TF2Torch(x), _TF2Torch(y)
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            return x, y
        
        if index//2*self.batch_size >= self._get_count(self._get_shorter()[0]):
            organism = self._get_shorter()[1]
        else:
            organism = ["human", "mouse"][index%2]

        for i in range(self.batch_size):
            x, y = get_data(organism)
            if i > 0:
                px, py = torch.cat([px,x], dim=0), torch.cat([py,y], dim=0)
            else:
                px, py = x, y
        return px, py, organism
    

def load_basenji2():
   return Basenji2Dataset("train"), Basenji2Dataset("test"), Basenji2Dataset("valid")
   config_gen = get_config("general")
   batch_size, num_workers = config_gen["batch_size"], config_gen["num_workers"]
   
   train_dataset = Basenji2Dataset("train")
   train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
   )

   test_dataset = Basenji2Dataset("test")
   test_loader = DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
   )

   valid_dataset = Basenji2Dataset("valid")
   valid_loader = DataLoader(
      dataset=valid_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
   )     

   return train_loader, test_loader, valid_loader