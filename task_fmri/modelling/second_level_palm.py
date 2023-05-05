import pandas as pd
import numpy as np
from decouple import config
import os
import shutil
import argparse
import glob
import re
import nilearn.image as img

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

class Contrast:

    '''
    A simple class to change pd.Dummies into a 1 and -1.

    Usage
    ----
    for column in df.columns:
            update = Contrast()
            df[column] = df[column].apply(update.contrast_genrator)

    '''
    def __init__(self) -> None:
        self.counter = 0
    def contrast_genrator(self, val):
        if val == 1 and self.counter == 1:
            return -1
        else:
            self.counter += val
            return val
              
def set_up_design_df(df: pd.DataFrame, subtractve=False) -> pd.DataFrame:
    
    '''
    Function to set up dataframe for design matrix.
    DataFrame must be set up as :

    T1          |T2
    ----------- |-------------
    path_to_scan  path_to_scan

    Parameters
    ----------
    df: pd.DataFrame 
        Dataframe of participants scans

    subtractive: bool 
        If true will set 0 in design matrix to -1. 
        Default False.
    Returns
    -------
    long_df: pd.Dataframe
            Dataframe in long form of group, time and intercept
    '''
    if subtractve == True:
        design_matrix_value = -1
    else:
        design_matrix_value = 0
    df['sub'] = df.index
    long_df = pd.melt(df, id_vars=['sub'], 
                      var_name='time_point', 
                      value_vars=['t1', 't2',], 
                      value_name='scans').sort_values(by=['sub'], ascending=True).reset_index(drop=True)
    long_df['group'] = long_df['scans'].apply(lambda participants: design_matrix_value if 'sub-G1' in participants or 'sub-B1' in participants else 1)
    long_df['time'] = long_df['time_point'].apply(lambda participants: design_matrix_value if 't1' in participants else 1)
    long_df['intercept'] = 1
    long_df = long_df.drop(long_df[long_df['sub'] == 75].index)

    return long_df

def create_design_matrix(path: str, subtractive=False, random_effects_subtractive=False) -> dict:
    
    '''
    Function to create a design matrix

    Parameters
    ----------
    path: str
        File path to csv of participants scans
    
    subtractive: bool 
        If true will set 0 in design matrix to -1. 
        Default False.

    random_effects_subtractive: bool 
        If true will set 1 in design matrix to -1. 
        Default False.
    Returns
    ------
    dict : dictionary object
        Dictionary of list of scans and
        design matrix
        
    '''

    participant_scans = pd.read_csv(f"{path}/1stlevel_location.csv")
    long_df = set_up_design_df(participant_scans, subtractive)
    scans = long_df['scans'].to_list()
    random_effects = pd.get_dummies(long_df['sub']).add_prefix('sub-')
    
    if random_effects_subtractive == True:
        for column in random_effects:
            update = Contrast()
            random_effects[column] = random_effects[column].apply(update.contrast_genrator)
    interaction_effect = long_df['time'] * long_df['group']
    design_matrix = pd.concat([long_df[['time']], interaction_effect.rename('interaction'),  
                               random_effects], axis=1) 
    
    return  {
        'scans': scans,
        'design_matrix': design_matrix
    }


def get_paths(task: str) -> dict:

    '''
    Function to return paths needed for workflow

    Parameters
    ---------
    task: str
        Name of task. Must be happy, fear or eft. 

    Returns
    -------
    dict: dictionary object
        dictionary of pathlike objects
    '''
    
    return {
        'base_dir': config(task),
        '2ndlevel_dir': os.path.join(config(task), '2ndlevel'),
    }

def create_design_files(design_matrix: pd.DataFrame, second_level_directory: str, scans: list) -> None:
    
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

    t_contrasts = np.vstack((
                    np.hstack(([1, 0], np.zeros(design_matrix.shape[1] -2))), # time
                    np.hstack(([0, 1], np.zeros(design_matrix.shape[1] -2))) # interaction
    ))
    t_contrasts = pd.DataFrame(t_contrasts)
    
    eb_data = {
        'block_one': [-1 for block in range(0, design_matrix.shape[0])],
        'within_perms': sorted([block for block in range(1, design_matrix.shape[1] -1)] + [block for block in range(1, design_matrix.shape[1] -1)]),
        'between_perms':  [1 if re.search(r'G1|G2', participant) else 2 for participant in scans],
        }

    eb_df = pd.DataFrame(eb_data)
    os.mkdir(f'{second_level_directory}/.designfiles')
    eb_df.to_csv(f'{second_level_directory}/.designfiles/eb_file.csv', index=False, header=False)
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
    design_files = glob.glob(os.path.join(paths['2ndlevel_dir'],".designfiles/*" ))
    mat_file = [file for file in design_files if 'design_matrix.csv' in file][0]
    con_one = [file for file in design_files if 't_contrasts.csv' in file][0]
    grp_file = [file for file in design_files if 'eb_file.csv' in file][0]
    palm_commnd = f"""
                    palm -i {copes_images}\
                               -m {mask_image}\
                               -n {perms}\
                               -d {mat_file}\
                               -t {con_one}\
                               -eb {grp_file}\
                               -o {results_directory}\
                               -T -logp -accel tail -nouncorrected -saveglm\
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
    paths = get_paths(flags['task'])

    # Creates design matrix and gets list of participants scans
    print('\nSetting up workflow\n')
    print('\tGetting participants scans and setting up design matrix\n')
    matrix_dict = create_design_matrix(paths['base_dir'], random_effects_subtractive=False, subtractive=False)
    
    # Creates and saves design files 
    print('\nCreating design files')
    create_design_files(matrix_dict['design_matrix'], paths['2ndlevel_dir'], matrix_dict['scans'])
    # Get all scans into same space and create a mask for palm
    creating_cope_and_mask(matrix_dict['scans'], paths['2ndlevel_dir'])

    # Makes results directory and defines path
    os.mkdir(os.path.join(paths['2ndlevel_dir'], 'mixed_model'))
    results_path = os.path.join(paths['2ndlevel_dir'], 'mixed_model')
    copes_images = os.path.join(paths['2ndlevel_dir'], 'copes_img.nii')
    mask_image = os.path.join(paths['2ndlevel_dir'], 'mask_img.nii')
    
    # Runs Palm
    print('\nStarting Palm now\n')
    run_palm(paths['2ndlevel_dir'], results_path, flags['perms'])
    
    # Moves files into results directory and deletes any working directories
    print('Cleaning up directory')
    [shutil.move(file, results_path) for file in glob.glob(os.path.join(paths['2ndlevel_dir'], '*')) if '.designfiles' not in file]
    os.system(f'gzip -f {results_path}/*nii')
    [shutil.move(file, results_path) for file in glob.glob(os.path.join(paths['2ndlevel_dir'], '.designfiles/*'))]
    shutil.rmtree(os.path.join(paths['2ndlevel_dir'], '.designfiles'))
    print('\nFinished')
    print('-'*100, '\n')