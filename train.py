import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import model
from data_loader import data_loader

NUM_EPOCHS = 10
BATCH_SIZE = 2



def train(num_block) :

    if(num_block <= 1) :
        print('Not enough block, terminate')
        return 
    
    generator = model.Generator(BATCH_SIZE, num_block)
    discriminator = model.Discriminator(num_block)

    z = torch.rand(100)

    for i in range(2, num_block + 1) :
        loader = iter(data_loader(i, BATCH_SIZE, root="./train_image", num_workers=0 ))
        
        real_image = next(loader)
        real_prediction = discriminator(real_image, i)

        fake_image = generator(z, i)
        fake_prediction = discriminator(fake_image, i)

        print(f"real = {real_prediction} \n fake = {fake_prediction}")
        


if __name__ == "__main__" :
    train(3)
 
    print('end of main')


