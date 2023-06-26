import pickle
from decouple import config
import re
import os
import pandas as pd
import glob

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
    
def get_image_paths(group: str, mean_images: bool =False) -> pd.DataFrame:
    '''
    Function to get imaging paths
    
    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN
    mean_images: bool
        bool to get mean images from the three tasks (default false)
    
    Returns
    -------
    pd.DataFrame: Dataframe
       Dataframe of path to images
    '''

    paths = glob.glob(os.path.join(config('eft'), '1stlevel', 'T1', f'sub-{group}*', 'ess_0004.nii'))
    
    if mean_images == True:
        paths = glob.glob(os.path.join(config('ml'), 'mean_task_images',  f'sub-{group}*.nii.gz'))
    
    return pd.DataFrame(
        data={
            'id': [re.findall(f'{group}...', participant)[0] for participant in paths],
            'eft_paths': paths,
            'happy_paths': [re.sub('eft', 'happy', path) for path in paths],
            'fear_paths': [re.sub('eft', 'fear', path) for path in paths]
            }
            ).sort_values(by='id')

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
        right_on='id'
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

def ados(group: str, mean_images: bool = False) -> pd.DataFrame:
    
    '''
    Wrapper function to  load and filter ados and image paths
    
    Parameters
    ----------
    group: str
        str of either G1 for HC or G2 for AN
    mean_images: bool
        bool to get mean images from the three tasks (default false)

    Returns
    -------
    pd.DataFrame: DataFrame
        Dataframe in long form
    '''

    ados_df = get_ados_df(group)
    image_paths = get_image_paths(group, mean_images)
    filtered_df = filter_ados_df(ados_df, image_paths)
    if mean_images == True:
        return filtered_df.drop(['happy_paths', 'fear_paths'], axis=1).rename(columns={'eft_paths': 'paths'})
    return organising_df_into_long_form(filtered_df)