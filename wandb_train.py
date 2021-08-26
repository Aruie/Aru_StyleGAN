


###########################################
# Make DataSet
###########################################

# #%%

# import wandb as wb



# #%%
# args = {
#     'project' : 'StyleGAN', 
#     'job_type' : 'raw-data', 
#     'tags' : ['Data Upload'],
#     'name' : 'Anime64'
# }



# with wb.init(**args) as run :
#     artifact = wb.Artifact(name = 'Anime64', 
#                             type = 'dataset',
#                             description = 'Anime 64 Data Make', )
#     artifact.add_file('../datasets/anime64.zip')
#     run.log_artifact(artifact)



#%%


import wandb

def dataLoad() : 

    args = {
        'project' : 'StyleGAN',
        'tags' : ['Data Download'],
        'name' : 'data_download'
    }

    with wandb.init(**args) as run :
        artifact = run.use_artifact('Anime64:latest', type='dataset')
        artifact_dir = artifact.download()

    with open('data_path.conf', 'w') as f :
        f.write(artifact_dir)



import os
import zipfile



#%%


data_path = './artifacts/Anime64:v0/anime64'


# %%


### from pytorch lightning tutorial
### https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

from model.Stylegan import MappingNet, Generator, Discriminator

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

from collections import OrderedDict

import numpy as np
from PIL import Image

#%%

class Anime64DataSet(LightningDataModule) :
    def __init__(self, 
                data_dir = './artifacts/Anime64:v0/anime64',
                batch_size = 4,
                num_workers = 1 ) :
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])

        self.dims = (3,64,64)
        self.num_classes = 0

    def prepare_data(self) : 
        
        file_list = os.listdir(self.data_dir)
        
        imgs = []

        for filename in file_list :
            img = Image.open(os.path.join(self.data_dir, filename))
            img = np.expand_dims( np.array(img).transpose(2,1,0), 0 )
            imgs.append(img)

        self.data = np.concatenate(imgs)
            
            

    def setup(self, stage=None) :
        pass
        # self.data = self.data
            

    def train_dataloader(self) :
        return DataLoader(self.data, )


        





#%%


class StyleGAN(LightningModule) :
    def __init__(self) :
        super().__init__()

        self.m = MappingNet()
        self.g = Generator(6)
        self.d = Discriminator(6)

        pass

    def forward(self, z) : 
        return self.g(z)

    def adversarial_loss(self, y_hat, y) : 
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx) :
        imgs, _ =  batch

        z = torch.randn(imgs.shape[0], 512 )
        z = z.type_as(imgs)


        # train generator
        if optimizer_idx == 0 :

            self.generated_images = self(z)

            sample_imgs = self.generated_images[:5]

            # log sample images
            grid = torchvision.utils.make_grid(sample_imgs) 
            self.logger.experiment.add_image('generated_images', grid, 0)

            valid = torch.ones(imgs.size(0), 1) 
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss( self.d(self(z)), valid)
            tqdm_dict = {'g_loss' : g_loss}
            output = OrderedDict({'loss' : g_loss,
                                'progress_var' : tqdm_dict,
                                'log' : tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1 :

            valid = torch.ones(imgs.size(0),1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.d(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.d(self.g(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss' : d_loss}
            output = OrderedDict({'loss' : d_loss,
                                'progress_var' : tqdm_dict,
                                'log' : tqdm_dict})
            return output

    def configure_optimizers(self) :
        lr = 0.001
        b1 = 0.5
        b2 = 0.999

        opt_g = torch.optim.Adam(self.g.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.d.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []






# %%

dm = Anime64DataSet()
model = StyleGAN()
trainer = Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
# %%
1
# %%
