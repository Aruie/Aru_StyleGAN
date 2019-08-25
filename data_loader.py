import os
from torch.utils import data
import torch
import PIL.Image as Image
import numpy as np

# 학습 데이터셋 정의
class TrainDataset(data.Dataset) :
    def __init__(self, step, path) :
        self.path = path
        self.samples = os.listdir(self.path)
        self.step = step

    def __getitem__(self, index) :
        
        sample = os.path.join(self.path, self.samples[index])
        image = Image.open(sample)

        # step별 resize
        width = 2 ** (self.step + 1)
        image = image.resize((width, width))
        image_array = np.array(image, dtype=np.float32).transpose(2,0,1) / 255.

        return image_array

    def __len__(self) :
        return len(self.samples)

# 샘플이 n개 이하일땐 그냥 셔플, n개 이상일땐 셔플 후 1000개 sampling
class MaxSampler(data.Sampler) :
    def __init__(self, data_source, max_sample) :
        self.data_source = data_source
        
        if len(self.data_source ) < max_sample :
            self.num_samples = len(self.data_source)
        else :
            self.num_samples = max_sample

    def __iter__(self) :
        return iter(torch.randperm(self.num_samples).tolist())

    def __len__(self) :
        return self.num_samples


def data_loader(step, batch_size, path, num_workers = 0) :
    dataset = TrainDataset(step, path)

    # 비복원 추출로 1000개 제한
    sampler = MaxSampler(dataset, 100)
    loader = data.DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            sampler = sampler)
    return loader

if __name__ == "__main__" :
    a = data_loader(1, 128, '../datasets/DogData/')
    #a = data_loader(1, 3, './train_image')
    len(a)

    for i, b in enumerate(a) :
        print(b.shape)