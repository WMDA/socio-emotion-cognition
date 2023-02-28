from decouple import config
import os
import pandas as pd

subject_list = ['sub-B1001']

for subject_id in subject_list:
    confounds_df = pd.read_csv(os.path.join(os.getenv('HOME'), subject_id) +
                        f'/func/{subject_id}_task-happy_desc-confounds_timeseries.tsv', sep='\t')

    movement_df = confounds_df[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    movement_df.to_csv(f'{os.getenv("HOME")}/{subject_id}/func/movement_regressors.txt', index=False)

