

#%%
import os


file_path = '../datasets/anime'
out_path = '../datasets/anime64'
data_list = os.listdir(file_path)




len(data_list)

# %%

from PIL import Image

h_list = []
w_list = []

for file in data_list :
    img = Image.open(os.path.join(file_path, file))
    
    w, h = img.size
    
    if w >= 64 :
        img.thumbnail((64,64), resample=Image.NEAREST)

        # os.path.join(out_path, file)
        img.save(os.path.join(out_path, file))
    

# %%


max(h_list), min(h_list), max(w_list), min(w_list)
# %%

sum(h_list) / len(h_list), sum(w_list) / len(w_list)


#%%
# %%



#%%
file_path = '../datasets/anime'
out_path = '../datasets/anime64'
data_list = os.listdir(out_path)




len(data_list)
# %%
from PIL import Image

h_list = []
w_list = []

for file in data_list :
    img = Image.open(os.path.join(out_path, file))
    
    w, h = img.size
    

    h_list.append(h)
    w_list.append(w)

    # if h != 64 :
    #     os.remove(os.path.join(out_path, file))


max(h_list), min(h_list), max(w_list), min(w_list)
# %%



