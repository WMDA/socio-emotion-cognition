import pandas as pd
import numpy as np
from decouple import config
import os
import shutil
import argparse
import glob
import re
import nilearn.image as img
import numpy as np
from itertools import chain

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
        '2ndlevel_dir': os.path.join(base_path, '2ndlevel')
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


def mean_imgs(subject_scans: pd.DataFrame) -> dict:
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

def create_design_files(design_matrix: pd.DataFrame, second_level_directory: str) -> None:
    
    '''
    Function to create design file. 
    
    The full model has T contrasts for group, time and interaction and a nested  
    exchangeable block structure which are all saved in a .desginfiles directory.

    Parameters
    ----------
    desgin_matrix: pd.DataFrame
        design matrix for palm

    second_level_directory: str
        str of path to test directory

    scans: list
        list of paths to participants scans

    Returns
    -------
    None
    '''

    t_contrasts = pd.Series(data=1)

    os.mkdir(f'{second_level_directory}/.designfiles')
    design_matrix.to_csv(f'{second_level_directory}/.designfiles/design_matrix.csv', index=False, header=False)
    t_contrasts.to_csv(f'{second_level_directory}/.designfiles/t_contrasts.csv', index=False, header=False)

def creating_cope_and_mask(scans: list, second_level_directory: str) -> None:
    
    '''
    Function to create cope and mask. 

    Concats scans into standard space for FSL (copes image). 

    Creates a group mask for randomise to use.

    Saves copes and mask as nii to directory. 

    Parameters
    ----------
    scans: list
        list of paths to participants scans

    second_level_directory: str
        str of path to test directory

    Returns
    -------
    None
    '''
    print('\nCreating copes and mask')
    print('\tConcating images into single space for FSL randomise')
    copes_concat = img.concat_imgs(scans, auto_resample=True)
    
    print(f'\t\tSaving combined nii file to {second_level_directory}\n')
    copes_concat.to_filename(os.path.join(second_level_directory, 'copes_img.nii'))

    print('\tCreating brainmask')
    mean_image = img.mean_img(scans)
    group_mask = img.binarize_img(mean_image)
    print(f'\t\tSaving mask nii file to {second_level_directory}\n')
    group_mask_sampled = img.resample_to_img(group_mask, copes_concat, interpolation='nearest')
    group_mask_sampled.to_filename(os.path.join(second_level_directory, "mask_img.nii"))


def run_palm(second_level_directory: str, results_directory: str, perms: int) -> None:

    '''
    Function to set up and then sys call PALM.

    Parameters
    ---------
    second_level_directory: str
        str of path to test directory

    results_directory: str
        str of path of where to save results

    perms: int
        Number of permutations for PALM to run
    '''

    copes_images = os.path.join(second_level_directory, 'copes_img.nii')
    mask_image = os.path.join(second_level_directory, 'mask_img.nii')
    design_files = glob.glob(os.path.join(second_level_directory, ".designfiles/*" ))
    mat_file = [file for file in design_files if 'design_matrix.csv' in file][0]
    con_one = [file for file in design_files if 't_contrasts.csv' in file][0]
    palm_commnd = f"""
                    palm -i {copes_images}\
                               -m {mask_image}\
                               -n {perms}\
                               -d {mat_file}\
                               -t {con_one}\
                               -o {results_directory}\
                               -T -logp -accel tail -nouncorrected\
                            """
    # -noranktest
    print('\nPalm command\n')
    print(palm_commnd)
    
    os.system(palm_commnd)

if __name__ == "__main__":

    # Intialise script by gettings flags and file paths
    print('\nStarting FSL PALM script\n')
    print('-'*100, '\n')
    flags = options()
    paths_location = paths(flags['task'])

    # Creates design matrix and gets list of participants scans
    print('\nSetting up workflow\n')
    print('\tGetting participants scans and setting up design matrix\n')
    participant_scans = subject_scans(paths_location['base_path'])
    scans = mean_imgs(participant_scans)
    design_matrix = create_desgin_matrix(scans)
    
    # Creates and saves design files 
    print('\nCreating design files')
    create_design_files(design_matrix, paths_location['2ndlevel_dir'])
    # Get all scans into same space and create a mask for palm
    list_of_images = list(chain.from_iterable(([scans['HC'], scans['AN']])))
    creating_cope_and_mask(list_of_images, paths_location['2ndlevel_dir'])

    # Makes results directory and defines path
    os.mkdir(os.path.join(paths_location['2ndlevel_dir'], 'group'))
    results_path = os.path.join(paths_location['2ndlevel_dir'], 'group')
    copes_images = os.path.join(paths_location['2ndlevel_dir'], 'copes_img.nii')
    mask_image = os.path.join(paths_location['2ndlevel_dir'], 'mask_img.nii')
    
    # Runs Palm
    print('\nStarting Palm now\n')
    run_palm(paths_location['2ndlevel_dir'], results_path, flags['perms'])
    
    # Moves files into results directory and deletes any working directories
    print('Cleaning up directory')
    [shutil.move(file, results_path) for file in glob.glob(os.path.join(paths_location['2ndlevel_dir'], '*')) if '.designfiles' not in file]
    os.system(f'gzip -f {results_path}/*nii')
    [shutil.move(file, results_path) for file in glob.glob(os.path.join(paths_location['2ndlevel_dir'], '.designfiles/*'))]
    shutil.rmtree(os.path.join(paths_location['2ndlevel_dir'], '.designfiles'))
    print('\nFinished')
    print('-'*100, '\n')