import os
from torch.utils import data
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



def data_loader(step, batch_size, path, num_workers = 0) :
    dataset = TrainDataset(step, path)



    ############# 수정
    if len(dataset) > 5 :
        sampler = data.RandomSampler(dataset, replacement = True, num_samples = 1000)
    else :
        sampler = data.RandomSampler(dataset)

    loader = data.DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            sampler = sampler)

    return loader

if __name__ == "__main__" :
    a = data_loader(1, 3, './train_image')
    for i, b in enumerate(a) :
        print(b.shape)