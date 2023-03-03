import pandas as pd
from decouple import config
import os
import shutil

import nilearn.image as img
from  nipype.interfaces import fsl
import nipype.pipeline.engine as pe
from nipype.interfaces.io import DataSink
from nipype import SelectFiles


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


def get_paths() -> dict:

    '''
    Function to return paths needed for workflow

    Parameters
    ---------
    None

    Returns
    -------
    dict : dictionary object
        dictionary of pathlike objects
    '''
    
    return {
        'base_dir': config('happy'),
        '2ndlevel_dir': os.path.join(config('happy'), '2ndlevel'),
    }


def setting_up_workflow(base_dir: str, test_directory: str) -> None:
    
    '''
    Function to set up workflow. Gets participants scans
    and concats into standard space for FSL randomise (copes image). 
    Creates a group mask for randomise to use. Saves copes and mask
    as nii.gz to directory. 

    Parameters
    ----------
    base_dir: str
        str of path to base directory

    test_directory: str
        str of path to test directory

    Returns
    -------
    None
    '''
    print('\nSetting up workflow\n')
    print('\tGetting participants scans design matrix\n')
    matrix_dict = create_design_matrix(base_dir)
    scans = matrix_dict['scans']
    scans = [scan for scan in scans if 'sub-B2' in scan]
    print('\tConcating images into single space for FSL randomise')
    copes_concat = img.concat_imgs(scans, auto_resample=True)
    
    print(f'\t\tSaving combined nii file to {test_directory}\n')
    copes_concat.to_filename(os.path.join(test_directory, 'copes_img.nii.gz'))

    print('\tCreating brainmask')
    mean_image = img.mean_img(scans)
    group_mask = img.binarize_img(mean_image)
    print(f'\t\tSaving mask nii file to {test_directory}\n')
    group_mask_sampled = img.resample_to_img(group_mask, copes_concat, interpolation='nearest')
    group_mask_sampled.to_filename(os.path.join(test_directory, "mask_img.nii.gz"))


def level2_workflow(test_directory: str, save_dir: str) -> None:

    '''
    Function to run FSL randomise in nipype.

    Parameters
    ----------

    test_directory: str
        str of path to test directory

    save_dir: str
        str of folder to save results

    Returns
    -------
    None
    '''

    level_2_analysis = pe.Workflow(name='randomise')
    level_2_analysis.base_dir = os.path.join(test_directory, 'workingdir')
    
    files = {
        'copes_image': os.path.join(test_directory, 'copes_img.nii.gz'),
        'mask_image': os.path.join(test_directory, 'mask_img.nii.gz')
            }
    selectfiles = pe.Node(SelectFiles(files,
                                   base_directory=test_directory,
                                   sort_filelist=True),
                        name='select_file')
    
    randomise = pe.Node(fsl.Randomise(one_sample_group_mean=True, 
                                      num_perm=50,
                                      vox_p_values=True,
                                      tfce=True,
                                      base_name='happy'), 
                                name='fsl_randomise')
    
    data_sink = pe.Node(DataSink(base_directory=test_directory,
                                 container=test_directory),
                        name='datasink') 

    level_2_analysis.connect([
        (selectfiles, randomise, [('copes_image', 'in_file')]),
        (selectfiles, randomise, [('mask_image', 'mask')]),         
        (randomise, data_sink, [#(f'f_corrected_p_files', f'{save_dir}.@f_corrected_p_files'),
                                        #('f_p_files', f'{save_dir}.@f_uncorrected'),
                                        #('fstat_files', f'{save_dir}.@fstat_files'),
                                        ('t_corrected_p_files', f'{save_dir}.@t_corrected_p_files'),
                                        #('t_p_files', f'{save_dir}.@t_uncorrected'),
                                        ('tstat_files', f'{save_dir}.@tstat_files')])
    ])
    
    print(f'Randomise workflow being set up\n')
    print('\tSaving Graph\n')
    level_2_analysis.write_graph(graph2use='colored', format='png', simple_form=True)
    
    print('Running Randomise\n')
    level_2_analysis.run()

if __name__ == "__main__":
    print('\nStarting FSL Randomise script\n')
    print('-'*100, '\n')
    fsl.FSLCommand.set_default_output_type('NIFTI')
    paths = get_paths()
    folder_to_save_results = '1sampleT'
    folder_path = os.path.join(paths['2ndlevel_dir'], folder_to_save_results)
    setting_up_workflow(paths['base_dir'], paths['2ndlevel_dir'])
    level2_workflow(paths['2ndlevel_dir'], folder_to_save_results)
    print('Cleaning up directory')
    os.system(f'gzip -f {folder_path}/*')
    shutil.move(os.path.join(paths['2ndlevel_dir'], 'mask_img.nii.gz'), folder_path)
    shutil.move(os.path.join(paths['2ndlevel_dir'], 'copes_img.nii.gz'), folder_path)
    shutil.move(os.path.join(paths['2ndlevel_dir'], 'workingdir', 'randomise', 'graph.png'), folder_path)
    shutil.rmtree(os.path.join(paths['2ndlevel_dir'], 'workingdir'))
    print('Finished')


