import sys
import os
from decouple import config
import os
import pandas as pd
import nilearn.glm.first_level as ngl
from nilearn import image as img

'''
Simple script to run beta series modelling
'''

def paths() -> dict:
    '''
    Function to return filepaths

    Parameters
    ----------
    None

    Returns
    -------
    dict of file paths
    '''    

    return {
        'base_dir': config('preprocessed_eft_2'),
        'csv_dir': config('bids_t2'),
        'mask_dir': config('preprocessed_eft_1'),
        'save_dir': os.path.join(config('eft'), 'spotlight', 'beta_regression', 'T1')
    }

def confounds(confounds_df: str) -> pd.DataFrame:
    '''
    Function to filter out fMRIPrep confounds df 
    to get confounds actually wanted

    Parameters
    ----------
    confounds_df: str
      file path to fMRIPrep confounds tsv

    Returns
    -------
    pd.DataFrame of filtered fMRIPrep confounds

    '''
    confounds_df = pd.read_csv(confounds_tsv, sep='\t')
    regres = confounds_df.filter(regex=("a_comp_cor.*")).to_dict(orient='series')
    regres.update([
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
    return pd.DataFrame(data=regres)

def save_beta_maps(beta_maps: dict, save_dir: str) - > None:

    '''
    Function to save beta maps for each condition

    Parameters
    ----------
    beta_maps: dict
        dictionary of beta maps

    save_dir: str
        filepath to where nii.gz images will be saved

    Returns
    -------
    None

    '''
    
    for key in beta_maps.keys():
        [scan.to_filename(f"{save_dir}/{key}_{number}.nii.gz") for number, scan in enumerate(beta_maps[key])]

def glm_1stlevel() -> ngl.FirstLevelModel:
    
    '''
    Function to define GLM

    Parameters
    ----------
    None

    Returns
    -------
    ngl.FirstLevelModel
    '''
    
    return ngl.FirstLevelModel(
        noise_model='ar1', 
        t_r=2,
        hrf_model='spm',
    )


def beta_maps(glm: ngl.FirstLevelModel,  events_df: pd.DataFrame, glm_events_df: pd.DataFrame) -> dict:
    
    '''
    Function to calculate beta maps

    Parameters
    ----------
    glm: ngl.FirstLevelModel
        Fitted first level glm model

    events_df: pd.DataFrame
        Dataframe of events

    glm_events_df: pd.DataFrame
        Updated events df with each event having its own distinct number

    
    Returns
    ------
    condition_beta_maps: dict
        dict of beta maps
    '''

    
    condition_beta_maps = {cond: [] for cond in events_df["trial_type"].unique()}
    trialwise_conditions = glm_events_df["trial_type"].unique()
    for condition in trialwise_conditions:
        beta_map = glm.compute_contrast(condition, output_type="effect_size")
        # Drop the trial number from the condition name to get the original name
        condition_name = condition.split("__")[0]
        condition_beta_maps[condition_name].append(beta_map)

    return condition_beta_maps
    
def glm_events(events_df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Function to update events_df so each event has a unqiue identifier
    in the trial_type column

    Parameters
    ---------
    events_df: pd.DataFrame
        Dataframe of events

    Returns
    -------
    glm_events_df: pd.DataFrame
        Updated events df
    '''
    glm_events_df = events_df.copy()
    conditions = glm_events_df["trial_type"].unique()
    condition_counter = {c: 0 for c in conditions}
    
    for i_trial, trial in glm_events_df.iterrows():
        trial_condition = trial["trial_type"]
        condition_counter[trial_condition] += 1
        trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
        glm_events_df.loc[i_trial, "trial_type"] = trial_name

    return glm_events_df


if __name__ == '__main__':
    subject = str(sys.argv[1])
    
    # To get all the files paths needed
    file_paths = paths()
    os.mkdir(file_paths['save_dir'])
    preprocessed = os.path.join(file_paths['base_dir'], 
                                subject, 
                                'func', 
                                f'{subject}_task-eft_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
    behavioural_csv = os.path.join(file_paths['csv_dir'], subject, 'func', f'{subject}_task-eft_events.tsv')
    confounds_tsv = os.path.join(file_paths['base_dir'], 
                                 subject, 
                                 'func', 
                                 f'{subject}_task-eft_desc-confounds_timeseries.tsv')
    
    # Read in nii image and confounds tsv
    fmri_file = img.load_img(preprocessed)
    confounds_df = confounds(confounds_tsv)
    
    # read in task csv
    events_df = pd.read_csv(behavioural_csv)
    events_df = events_df.rename(columns={'Condition': 'trial_type'})
    
    # Create the glm_events_df
    glm_events_df = glm_events(events_df)

    # Define the glm
    glm = ngl.FirstLevelModel(
        noise_model='ar1', 
        t_r=2,
        hrf_model='spm',
    )
    
    # Fit the glm
    glm.fit(fmri_file, glm_events_df.iloc[0:,1:4], confounds=confounds_df.fillna(0))
    
    # Caulated beta maps
    beta_maps_dict = beta_maps(glm, events_df, glm_events_df)    
    
    # Save beta maps
    save_beta_maps(beta_maps_dict, file_paths['save_dir'])