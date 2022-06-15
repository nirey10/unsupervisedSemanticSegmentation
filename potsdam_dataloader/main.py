from dataloader import PotsdamDataset
from torch.utils import data
import tqdm
import numpy as np

batch_size = 16

potsdam_dataset = PotsdamDataset(data_root='../../datasets/potsdam_dataset/')
loader = data.DataLoader(potsdam_dataset, batch_size=batch_size)

for i, sample in enumerate(loader):
    image_ids, images, labels = sample