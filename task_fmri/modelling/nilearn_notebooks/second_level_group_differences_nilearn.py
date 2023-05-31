import pandas as pd
from decouple import config
import numpy as np
import os
import nilearn.image as img
from nilearn.glm.second_level import non_parametric_inference
import nibabel
import argparse

def options() -> dict:

    '''
    Function to accept accept command line flags.
    Needs -t for task name and -p for number of 
    permutations

    Parameters
    ---------
    None

    Returns
    -------
    dict: dictionary object
        Dictionary of task and number of 
        permuations
    '''
        
    args= argparse.ArgumentParser()
    args.add_argument('-t', '--task', 
                      dest='task', 
                      help='Task name. Either happy, eft or fear')
    args.add_argument('-p', '--perms',
                      dest='perms',
                      help='number of permutations to run')
    return vars(args.parse_args())

def paths(task: str) -> dict:
    '''
    Function to return paths to save and load images.

    Parameters
    ----------
    task: str
       str of happy, fear, eft.

    Returns
    -------
    dict of paths
    '''
    base_path = config(task)

    return {
        'base_path': base_path,
        'mixed_model': os.path.join(base_path, '2ndlevel', 'mixed_model')
    }


def subject_scans(base_path: str) -> pd.DataFrame:

    '''
    Function to load csv of subjects scans.
    Will remove one subject who doesn't have T1 scan.

    Parameters
    ----------
    base_path: str
       absolute path to task directory

    Returns
    -------
    subject_scans_df: pd.DataFrame
        csv of subjects scans locations 
    '''
    subject_scans_df = pd.read_csv(f"{base_path}/1stlevel_location.csv")
    subject_scans_df = subject_scans_df.drop(subject_scans_df[subject_scans_df['t1'] == 75].index)
    return subject_scans_df


def create_desgin_matrix(subjects_scans: dict) -> pd.DataFrame:
    '''
    Function to create a singe design matrix of group

    Parameters
    ----------
    subjects_scans: dict,
        dictionary of subject images with keys of group.

    Returns
    -------
    design matrix: pd.DataFrame,
       (92 x 1) design matrix of -1 and 1.
    '''
    
    return pd.DataFrame(data={'Group': np.hstack((-np.ones(len(subjects_scans['HC'])), np.ones(len(subjects_scans['AN']))))})


def mean_img(subject_scans: pd.DataFrame) -> dict:
    '''
    Function to get the mean image from the two time points

    Parameters
    ----------
    subject_scans: pd.DataFrame.
        Dataframe of location of subjects scans of T1, T2 

    Returns
    -------
    subjects_mean_images: dict
       dictionary of mean images 

    '''

    subjects_mean_images = {
    'HC' : [],
    'AN' : []
    }

    for subject in range(0, subject_scans.shape[0]):
        try:
            t1_image = img.load_img(subject_scans['t1'].iloc[subject])
            t2_image = img.load_img(subject_scans['t2'].iloc[subject])
            mean_img = img.mean_img([t1_image, t2_image])

        except Exception as e:
            print(e)
            continue
    
        if 'G1' in subject_scans['t1'].iloc[subject]:
            subjects_mean_images['HC'].append(mean_img)
        else:
            subjects_mean_images['AN'].append(mean_img)

    return subjects_mean_images


def ols(subjects_to_analyse: list, 
        design_matrix: pd.DataFrame, 
        masks_2ndlevel: nibabel.nifti1.Nifti1Image,
        perm: int) -> dict:

    '''
    Function to run nilearn permutated ols.

    Parameters
    ----------
    subjects_to_analyse: list
        list of nibabel.nifti1.Nifti1Image scans 

    design_matrix: pd.DataFrame
        (92 x 1) design matrix of group

    mask_2ndlevel: nibabel.nifti1.Nifti1Image
        mask of 1st level inputs

    perm: int
        Number of permutations

    Returns
    -------
    dictionary of nibabel.nifti1.Nifti1Image
    '''
    return non_parametric_inference(
    second_level_input=subjects_to_analyse,
    design_matrix=design_matrix,
    second_level_contrast="Group",
    mask=masks_2ndlevel,
    model_intercept=True,
    n_perm=int(perm),
    n_jobs=6,
    tfce=True,
    verbose=3
    )

if __name__ == "__main__":
    print('Starting up permutated ols for group differences')
    flags = options()
    path = paths(flags['task'])
    scans_location = subject_scans(path['base_path'])
    mean_images = mean_img(scans_location)
    design_matrix = create_desgin_matrix(mean_images)
    mask = img.load_img(os.path.join(path['mixed_model'], 'mask_img.nii.gz' ))
    print(f'Running OLS with {flags["perms"]} permutations for {flags["task"]} task')
    subjects_to_analyse = mean_images['HC'] + mean_images['AN']
    group_diff = ols(subjects_to_analyse, design_matrix, mask, flags["perms"])
    print(f'Saving scans to {path["mixed_model"]}')
    group_diff['logp_max_tfce'].to_filename(f'{path["mixed_model"]}/tfce_fwep_group.nii.gz')
    group_diff['tfce'].to_filename(f'{path["mixed_model"]}/tfce_tstat_group.nii.gz')
    group_diff['t'].to_filename(f'{path["mixed_model"]}/vox_tstat_group.nii.gz')
    group_diff['logp_max_t'].to_filename(f'{path["mixed_model"]}/vox_fwep_group.nii.gz')