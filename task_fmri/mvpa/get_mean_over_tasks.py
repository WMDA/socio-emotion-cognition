from decouple import config
import pandas as pd
import glob
import os
import re
from nilearn import image as img
from mvpa_functions import ados


if __name__ == "__main__":
    tv_l1_path = config('ml')
    base_path = config('eft')
    test_eft = ados('G2', test_train=None, directory='eft')
    subject_scans_df = pd.read_csv(f"{base_path}/1stlevel_location.csv")
    subject_scans_df = subject_scans_df.drop(subject_scans_df[subject_scans_df['t1'] == 75].index)
    subject_scans_df['G-Number'] = subject_scans_df['t1'].apply(lambda part: re.findall('G....', part)[0]).reset_index(drop=True)
    subject_scans_df['B-Number'] = subject_scans_df['t2'].apply(lambda part: re.findall(r'B\d...', part)[0]).reset_index(drop=True)
    com = pd.merge(test_eft['G-Number'], subject_scans_df, on='G-Number')
    beta_images_paths = pd.DataFrame(
        data={
            'id': com['G-Number'],
            'eft_paths_t1': com['t1'],
            'eft_paths_t2': com['t2'],
            'happy_paths_t1': [re.sub('eft', 'happy', path) for path in  com['t1']],
            'happy_paths_t2': [re.sub('eft', 'happy', path) for path in  com['t2']],
            'fear_paths_t1': [re.sub('eft', 'fear', path) for path in  com['t1']],
            'fear_paths_t2': [re.sub('eft', 'fear', path) for path in  com['t2']]
            }
            ).sort_values(by='id')
    
    test_dir = os.listdir(os.path.join(tv_l1_path, 'mean_task_images', 'test', 't1'))
    for row in beta_images_paths.iterrows():
        print(f'Working on subject {row[1]["id"]}')
        eft_t1 = img.load_img(row[1]['eft_paths_t1'])
        eft_t2 = img.load_img(row[1]['eft_paths_t2'])
        happy_t1 = img.load_img(row[1]['happy_paths_t1'])
        happy_t2 = img.load_img(row[1]['happy_paths_t2'])
        fear_t1 = img.load_img(row[1]['fear_paths_t1'])
        fear_t2 = img.load_img(row[1]['fear_paths_t2'])
        combined_mean = img.mean_img([eft_t1, eft_t2, happy_t1, happy_t2, fear_t1, fear_t2])
        t2 = img.mean_img([eft_t2, happy_t2, fear_t2])
        directory = "train"
        if f"sub-{row[1]['id']}.nii.gz" in test_dir:
            directory = "test"
        t2.to_filename(f'{tv_l1_path}/mean_task_images/{directory}/t2/sub-{row[1]["id"]}.nii.gz')
        combined_mean.to_filename(f'{tv_l1_path}/mean_task_images/{directory}/combined/sub-{row[1]["id"]}.nii.gz')