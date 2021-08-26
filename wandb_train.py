


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