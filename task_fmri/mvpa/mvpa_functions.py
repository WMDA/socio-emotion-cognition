import pickle
from decouple import config
import re
import os
import pandas as pd
import glob
import sys

def save_pickle(name: str, object_to_pickle: object) -> None:

    '''
    Function to save an object as pickle file.
   
    Parameters
    ----------
    name: str 
        str of name of file. Include full path
    object_to_pickle: object 
        object to save as pickle file
    
    Returns
    -------
    None
    '''

    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(object_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name_of_pickle_object: str) -> object:

    '''
    Function to load pickle object.

    Parameters
    ----------
    name_of_pickle_object: str 
        name of object to be loaded. 
        Doesn't need extension however needs full path

    Returns
    -------
    unpickled obect
    '''


    try: 
        with open(f'{name_of_pickle_object}.pickle', 'rb') as handle:
            return pickle.load(handle)
    except Exception:
        print('Unable to load pickle object')



def get_ados_df(group: str):
    '''
    Function to get ados values for participants

    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN
    
    Returns
    ------
    pd.DataFrame: DataFrame
        DataFrame of filtered ados values by group
    '''

    ados = pd.read_csv(f'{config("ml")}/behavioural_results.csv')[['G-Number', 'ADOS_Communication',
           'ADOS_Interaction', 'ADOS_com_soc', 'ADOS_Creativity',
           'ADOS_sterotyped_and_repetititve']].dropna()
    
    return ados[ados['G-Number'].str.contains(group)].reset_index(drop=True)

def build_df(group: str, paths: list) -> dict:
    
    '''
    fucntion to build the df of image paramaters
    
    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN  
    paths: list
        list of paths to images
    
    Returns
    -------
    pd.DataFrame: Dataframe
       Dataframe of path to images
    '''

    return pd.DataFrame(
        data={
            'id': [re.findall(f'{group}...', participant)[0] for participant in paths],
            'eft_paths': paths,
            'happy_paths': [re.sub('eft', 'happy', path) for path in paths],
            'fear_paths': [re.sub('eft', 'fear', path) for path in paths]
            }
            ).sort_values(by='id')

def get_image_paths(group: str, 
                    test_train: str,
                    directory: str,
                    mean_images: bool = True) -> pd.DataFrame:
    '''
    Function to get imaging paths
    
    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN    
    test_train: str
        string object to decide if images are for testing or training
    directory: str
        string of either eft, t1 or combined of where to get the images from
    mean_images: bool
        bool to get mean images from the three tasks (default True)

    
    Returns
    -------
    pd.DataFrame: Dataframe
       Dataframe of path to images
    '''

    if directory == 'eft':    
        paths = glob.glob(os.path.join(config('eft'), '1stlevel', 'T1', f'sub-{group}*', 'ess_0004.nii'))
        return build_df(group, paths).drop(['happy_paths', 'fear_paths'], axis=1)
    
    if directory == 't1':
        paths = glob.glob(os.path.join(config('ml'), 'mean_task_images', test_train, 't1', f'sub-{group}*.nii.gz'))
        if 'G1' in group:
            paths = glob.glob(os.path.join(config('ml'), 'mean_task_images', 'hc', f'sub-{group}*.nii.gz'))
        return build_df(group, paths)
    
    if directory == 'combined':
        paths = glob.glob(os.path.join(config('ml'), 'mean_task_images', test_train, 'combined', f'sub-{group}*.nii.gz'))
        return build_df(group, paths)

    if directory == 't2':
        paths = glob.glob(os.path.join(config('ml'), 'mean_task_images', test_train, 't2', f'sub-{group}*.nii.gz'))
        return build_df(group, paths)

            
def filter_ados_df(ados: pd.DataFrame, beta_images_paths: pd.DataFrame) -> pd.DataFrame:
    '''
    Functionto filter the ADOS and imaging paths to 
    subjects that actually have both

    Parameters
    ----------
    beta_images_paths: pd.DataFrame
       Dataframe of path to images
    
    ados: pd.DataFrame
        Dataframe of  ADOS values
    Returns
    -------
    pd.DataFrame: DataFrame
        DataFrame of filtered ados values by group
    '''

    particpants_with_ados = ados[ados['G-Number'].isin(beta_images_paths['id'])].reset_index(drop=True)
    return pd.merge(
        left=particpants_with_ados, 
        right=beta_images_paths, 
        left_on='G-Number', 
        right_on='id',
        ).drop('id', axis=1).sort_values(by='G-Number')
    
def organising_df_into_long_form(ados_df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Function to organise dataframe into long form

    Parameters
    ----------
    ados_df: pd.DataFrame
        DataFrame of filtered ados values by group
    
    Returns
    -------
    pd.DataFrame: DataFrame
        Dataframe in long form

    '''
    ados_df_long = pd.melt(ados_df, 
                           value_vars=['eft_paths', 'happy_paths', 'fear_paths'], 
                           id_vars='G-Number').sort_values(by='G-Number').drop('variable', axis=1).reset_index(drop=True)
    
    return pd.concat((ados_df.loc[ados_df.index.repeat(3)].reset_index(drop=True), 
                        ados_df_long['value']), 
                        axis=1
                        ).drop(['eft_paths', 'happy_paths', 'fear_paths'], axis=1).rename(columns={'value': 'paths'})

def check_directory(directory: str) -> None:
    '''
    Function to check that directory input is correct
    Exits with sys.exit(1) if not

    Parameters
    ----------
    directory: str
        string of either eft, t1 or combined of where to get the images from
    Returns
    -------
    None
    '''
    values = ['eft', 't1', 't2' , 'combined']
    if directory in values:
            pass
    else:
        print('Invalid input, please enter either, eft, t1, t2 or combined')
        sys.exit(1)

def ados(group: str, test_train: str, directory: str, mean_images: bool = True) -> pd.DataFrame:
    
    '''
    Wrapper function to  load and filter ados and image paths
    
    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN
    test_train: str
        string object to decide if images are for testing or training
    directory: str
        string of either eft, t1, t2 or combined of where to get the images from
    mean_images: bool
        bool to get mean images from the three tasks (default True)

    Returns
    -------
    pd.DataFrame: DataFrame
        Dataframe in long form
    '''
    check_directory(directory)
    ados_df = get_ados_df(group)
    image_paths = get_image_paths(group, test_train, directory, mean_images)
    filtered_df = filter_ados_df(ados_df, image_paths)
    if directory == 'eft':
        return filtered_df
    if mean_images == True:
        return filtered_df.drop(['happy_paths', 'fear_paths'], axis=1).rename(columns={'eft_paths': 'paths'})
    return organising_df_into_long_form(filtered_df)