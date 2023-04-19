from decouple import config
import os
import pandas as pd

subject_list = ['sub-B2999']
dfs_events = os.path.join(config('raw_data'), 'bids_t2')
events_df = pd.read_csv(dfs_events + f'/{subject_list[0]}/func/{subject_list[0]}_task-eft_events.tsv')#, sep='\s')
file_path = config('preprocessed_fear_1')

confounds_df = pd.read_csv(
                        f'{file_path}{subject_list[0]}/func/{subject_list[0]}_task-fear_desc-confounds_timeseries.tsv', sep='\t')

acompor_dic = confounds_df.filter(regex=("a_comp_cor.*")).to_dict(orient='series')
acompor_dic.update([
        ('trans_x', confounds_df['trans_x']),
        ('trans_y', confounds_df['trans_y']),
        ('trans_z', confounds_df['trans_z']),
        ('rot_x', confounds_df['rot_x']),
        ('rot_y', confounds_df['rot_y']),
        ('rot_z', confounds_df['rot_z']),                   
        ('trans_x_derivative1', confounds_df['trans_x_derivative1']),
        ('trans_y_derivative1', confounds_df['trans_y_derivative1']),
        ('trans_z_derivative1', confounds_df['trans_z_derivative1']),
        ('rot_x_derivative1', confounds_df['rot_x_derivative1']),
        ('rot_y_derivative1', confounds_df['rot_y_derivative1']),
        ('rot_z_derivative1', confounds_df['rot_z_derivative1']), 
        ('trans_x_power2', confounds_df['trans_x_power2']),
        ('trans_y_power2', confounds_df['trans_y_power2']),
        ('trans_z_power2', confounds_df['trans_z_power2']),
        ('rot_x_power2', confounds_df['rot_x_power2']),
        ('rot_y_power2', confounds_df['rot_y_power2']),
        ('rot_z_power2', confounds_df['rot_z_power2']), 
        ('trans_x_derivative1_power2', confounds_df['trans_x_derivative1_power2']),
        ('trans_y_derivative1_power2', confounds_df['trans_y_derivative1_power2']),
        ('trans_z_derivative1_power2', confounds_df['trans_z_derivative1_power2']),
        ('rot_x_derivative1_power2', confounds_df['rot_x_derivative1_power2']),
        ('rot_y_derivative1_power2', confounds_df['rot_y_derivative1_power2']),
        ('rot_z_derivative1_power2', confounds_df['rot_z_derivative1_power2']),
        ])
