import pandas as pd
from fNeuro.MVPA.mvpa_functions import ados,get_ados_df
from decouple import config
import os
import re
import shutil


if __name__ == '__main__':
    path = config('ml')
    unfiltered_df = get_ados_df('G2')
    ados_df = ados('G2', mean_images=True, test_train=None).reset_index(drop=True)
    train_df = ados_df.sample(frac=0.8, random_state=3)
    train_df['paths'] = train_df['paths'].apply(lambda path: re.sub('/sub', '/train/sub', path))
    test_df = ados_df.drop(train_df.index)
    test_df['paths'] = test_df['paths'].apply(lambda path: re.sub('/sub', '/test/sub', path))
    train_df.to_csv(os.path.join(path, 'mean_task_images', 'train', 'ados_train.csv'))
    test_df.to_csv(os.path.join(path, 'mean_task_images', 'test', 'ados_test.csv'))
    print(train_df.shape)
    print(test_df.shape)
    [shutil.move(os.path.join(path, 
                              "mean_task_images", 
                              f'sub-{sub}.nii.gz'), 
                os.path.join(path, 'mean_task_images', 'train')) for sub in train_df['G-Number']]
    
    [shutil.move(os.path.join(path, "mean_task_images", 
                              f'sub-{sub}.nii.gz'), 
                os.path.join(path, 'mean_task_images', 'test')) for sub in test_df['G-Number']]

