import pandas as pd
import numpy as np
from decouple import config
import os
import shutil
import argparse
import glob
import nilearn.image as img
from  nipype.interfaces import fsl

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


def set_up_design_df(df: pd.DataFrame) -> pd.DataFrame:
    
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

    Returns
    -------
    long_df: pd.Dataframe
            Dataframe in long form of group, time and intercept
    '''

    df['sub'] = df.index
    long_df = pd.melt(df, id_vars=['sub'], 
                      var_name='time_point', 
                      value_vars=['t1', 't2',], 
                      value_name='scans').sort_values(by=['sub'], ascending=True).reset_index(drop=True)
    long_df['group'] = long_df['scans'].apply(lambda participants: -1 if 'sub-G1' in participants or 'sub-B1' in participants else 1)
    long_df['time'] = long_df['time_point'].apply(lambda participants: -1 if 't1' in participants else 1)
    long_df['intercept'] = 1
    long_df = long_df.drop(long_df[long_df['sub'] == 75].index)

    return long_df

def create_design_matrix(path: str) -> dict:
    
    '''
    Function to create a design matrix

    Parameters
    ----------
    path: str
        File path to csv of participants scans
    out_directory: str
        Directory to store design matrix

    Returns
    ------
    dict : dictionary object
        Dictionary of list of scans and
        design matrix
        
    '''

    participant_scans = pd.read_csv(f"{path}/1stlevel_location.csv")
    long_df = set_up_design_df(participant_scans)
    scans = long_df['scans'].to_list()
    random_effects = pd.get_dummies(long_df['sub']).add_prefix('sub-')
    interaction_effect = long_df['time'] * long_df['group']
    design_matrix = pd.concat([long_df[['intercept', 'group', 'time']], 
                               interaction_effect.rename('interaction'), 
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
        Name of task. Must be
        happy, fear or eft 

    Returns
    -------
    dict: dictionary object
        dictionary of pathlike objects
    '''
    
    return {
        'base_dir': config(task),
        '2ndlevel_dir': os.path.join(config(task), '2ndlevel'),
    }

def text_2_vest(in_file_name: str, out_file_name: str) -> None:
    # TODO this function needs testing out on the grid servers
    '''
    Function wrapper around Text2Vest to create FSL 
    mat, grp, con and fts files

    Parameters
    ----------
    in_file_name: str
        Path of txt file to be converted into FSL file

    out_file_name: str
        Name in form of path of the FSL file

    Returns
    -------
    None
    '''
    t2v = fsl.Text2Vest()
    t2v.inputs.in_file = in_file_name
    t2v.inputs.out_file = out_file_name
    print(t2v.cmdline)
    res = t2v.run() 

def create_design_files(desgin_matrix: pd.DataFrame, second_level_directory: str) -> None:
    
    '''
    Function to create design file. 
    
    The full model has T contrasts for group, time and interaction, a single F contrast and a nested  
    exchangeable block structure which are all saved in a .desginfiles directory
    initially as .txt files.

    Parameters
    ----------
    desgin_matrix: pd.DataFrame
        design matrix for palm

    second_level_directory: str
        str of path to test directory

    Returns
    -------
    None
    '''

    t_contrasts = np.vstack((
                        np.hstack(([1, 0, 0], np.zeros(desgin_matrix.shape[1] -3))), 
                        np.hstack(([0, 1, 0], np.zeros(desgin_matrix.shape[1] -3))), 
                        np.hstack(([0, 0, 1], np.zeros(desgin_matrix.shape[1] -3)))
    ))

    f_contrast = np.hstack(([1, 1, 1], np.zeros(desgin_matrix.shape[1] -3)))
    
    # TODO this needs changing for the hiearchical structure 
    exchange_blocks = sorted([block for block in range(1, desgin_matrix.shape[1] - 2)] + [block for block in range(1, desgin_matrix.shape[1] - 2)])

    os.mkdir(f'{second_level_directory}/.designfiles')
    np.savetxt(f'{second_level_directory}/.designfiles/design_mat.txt', desgin_matrix.values, fmt='%d')
    np.savetxt(f'{second_level_directory}/.designfiles/design_fts.txt', f_contrast, fmt='%d')
    np.savetxt(f'{second_level_directory}/.designfiles/design_grp.txt', exchange_blocks, fmt='%d')
    np.savetxt(f'{second_level_directory}/.designfiles/design_con.txt', t_contrasts, fmt='%d')

def set_up_deign_files(desgin_matrix: pd.DataFrame, second_level_directory: str) -> None:
   
    '''
    Function to create design files in the FSL VEST fomat.

    Parameters
    ----------
    desgin_matrix: pd.DataFrame
        design matrix for palm

    second_level_directory: str
        str of path to test directory

    Returns
    -------
    None

    '''
    create_design_files(desgin_matrix, second_level_directory)
    design = glob.glob(f'{second_level_directory}/.designfiles/*.txt')

    for design_file in design:
        out_file_name = re.sub(r'.txt|_', '.', design_file).rstrip('.')
        text_2_vest(design_file, out_file_name)

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




def run_palm(second_level_directory: str, results_directory: str, perms: int):

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
    mat_file = [file for file in design_files if 'design.mat' in file][0]
    con_file = [file for file in design_files if 'design.con' in file][0]
    fts_file = [file for file in design_files if 'design.fts' in file][0]
    grp_file = [file for file in design_files if 'design.grp' in file][0]
    print('\nPalm command\n')
    print(f'palm -i {copes_images}\
               -m {mask_image}\
               -n {perms}\
               -d {mat_file}\
               -t {con_file}\
               -f {fts_file}\
               -eb {grp_file}\
               -o {results_directory}\
               -vg auto\
               -ee -T  -logp'
          )
    
    os.system(f'palm -i {copes_images}\
               -m {mask_image}\
               -n {perms}\
               -d {mat_file}\
               -t {con_file}\
               -f {fts_file}\
               -eb {grp_file}\
               -o {results_directory}\
               -vg auto\
               -ee -T  -logp')

if __name__ == "__main__":

    # Intialise script by gettings flags and file paths
    print('\nStarting FSL PALM script\n')
    print('-'*100, '\n')
    flags = options()
    paths = get_paths(flags['task'])
    
    # Creates design matrix and gets list of participants scans
    print('\nSetting up workflow\n')
    print('\tGetting participants scans and setting up design matrix\n')
    matrix_dict = create_design_matrix(paths['base_dir'])
    
    # Creates and saves design files 
    print('\nCreating design files')
    set_up_deign_files(matrix_dict['design_matrix'], paths['2ndlevel_dir'])

    # Get all scans into same space and create a mask for palm
    creating_cope_and_mask(matrix_dict['scans'], paths['2ndlevel_dir'])

    # Makes results directory and defines path
    os.mkdir(paths['2ndlevel_dir']'/hierarchical_model')
    results_path = os.path.join(paths['2ndlevel_dir'], 'hierarchical_model')
    
    copes_images = os.path.join(paths['2ndlevel_dir'], 'copes_img.nii')
    mask_image = os.path.join(paths['2ndlevel_dir'], 'mask_img.nii')

    # Runs Palm
    print('\nStarting Palm now\n')
    run_plam(paths['2ndlevel_dir'], results_path, flags['perms'])
    
    # Moves files into results directory and deletes any working directories
    print('Cleaning up directory')
    shutil.move(os.path.join(paths['2ndlevel_dir'], 'mask_img.nii'), results_path)
    shutil.move(os.path.join(paths['2ndlevel_dir'], 'copes_img.nii'), results_path)
    os.system(f'gzip -f {results_path}/*')
    shutil.move(glob.glob(os.path.join(paths['2ndlevel_dir'], '.designfiles/*',), results_path))
    shutil.rmtree(os.path.join(paths['2ndlevel_dir'], '.designfiles'))
    print('\nFinished')
    print('-'*100, '\n')


