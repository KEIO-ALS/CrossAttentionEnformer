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
        "Basenji", f'data_{organism}_tfrecords_{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

class Basenji2_Datasets():
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
  

class Basenji2Loader():
    def __init__(self, subset) -> None:
        self.subset = subset
        human_dataloader_list = Basenji2_Datasets("human", subset)
        mouse_dataloader_list = Basenji2_Datasets("mouse", subset)

        self.iters = {
           "human": self._get_iter(human_dataloader_list),
           "mouse": self._get_iter(mouse_dataloader_list),
        }

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
       return self._get_count("human") + self._get_count("mouse")

    def __getitem__(self, index):
        if index//2 >= self._get_count(self._get_shorter()[0]):
            organism = self._get_shorter()[1]
        else:
            organism = ["human", "mouse"][index%2]
        x, y = next(self.iters[organism])
        return _TF2Torch(x), _TF2Torch(y), organism

        




# CIFAR10データセットをロードする関数 -> (trainloader, testloader)
def load_cifar10():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")
    
    # データの前処理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 訓練データセット
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # テストデータセット
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader

if __name__ == "__main__":
    trainloader = Basenji2Loader("test")
    pbar = tqdm(total=len(trainloader))
    for i, data in enumerate(trainloader, 0):
       x, y = data
       pbar.update()
    pbar.close()