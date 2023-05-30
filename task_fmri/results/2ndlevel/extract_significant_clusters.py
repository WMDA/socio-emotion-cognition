from decouple import config
import os
import numpy as np
import second_level_functions as slf
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #Filterout all the nilearn user warnings

tasks = ['happy', 'fear']#, 'eft']

for task in tasks:
    
    print(f'Working on extracting significant clusters from {task}')
    path = os.path.join(config(task), '2ndlevel', 'mixed_model')
    save_results = os.path.join(config(task), '2ndlevel')
    
    images = slf.get_images(path)
    contrasts = slf.contrast_imgs(images['t_stat'], images['pvals'])
    
    contrasts_list = [contrasts['group'], contrasts['time'], contrasts['interaction']]
    threshold_pval = -np.log10(0.05)
    
    for contrast in contrasts_list:
        slf.Results_table(
            contrast=contrast['pvals'],
            stat_threshold_val=threshold_pval,
            cluster_threshold_val=0,
            contrast_name=contrast['contrast_name'],
            output_directory=save_results,
            graph_name=contrast['graph_title']
        )