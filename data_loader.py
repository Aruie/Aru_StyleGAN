import os
from torch.utils import data
import PIL.Image as Image
import numpy as np

class TrainDataset(data.Dataset) :
    def __init__(self, step, path) :
        self.path = path
        self.samples = os.listdir(self.path)
        self.step = step

    def __getitem__(self, index) :
        
        sample = os.path.join(self.path, self.samples[index])
        image = Image.open(sample)

        width = 2 ** (self.step + 1)
        image = image.resize((width, width))
        image_array = np.array(image, dtype=np.float32).transpose(2,0,1) / 255.

        return image_array

    def __len__(self) :
        return len(self.samples)



def data_loader(step, batch_size, path, num_workers = 0) :
    dataset = TrainDataset(step, path)

    loader = data.DataLoader(dataset = dataset,
                             batch_size = batch_size,
                             shuffle = True,
                             num_workers = num_workers)
    return loader



if __name__ == "__main__" :
    a = data_loader(1, 4, './train_image')
    b = next(iter(a))
    print(b.shape)