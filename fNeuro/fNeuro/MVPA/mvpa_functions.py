import pickle
from decouple import config
import re
import os
import pandas as pd
import glob
import sys
from nilearn.maskers import NiftiSpheresMasker
import nilearn
import numpy as np
from fNeuro.second_level.second_level_functions import bayesian_cluster_info

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

def extract(coords, beta_img, t_score_img) -> pd.DataFrame:
    
    '''
    Function to extract beta wieghts and t scores
    from images at set co-ordinates

    Parameters
    ----------
    coords: tuple,
        tuple of X, Y, Z MNI co-ordinates
    beta_img: image,
        image of beta weights
    t_score: image,
        image of tscores
    
    Returns
    -------
    pd.DataFrame: DataFrame
        DataFrame of beta and t scores
    '''
    
    masker = NiftiSpheresMasker(
        (coords.values),
    )
    betas_frem = masker.fit_transform(beta_img)
    tscore_frem = masker.fit_transform(t_score_img)
    return pd.DataFrame({'beta': betas_frem.ravel(), 't_score': tscore_frem.ravel()})

def fetch_atlas(atlas: str) -> dict:
    '''
    Function to get atlas BUNCH

    Parameters
    ----------
    atlas: str
        str of ho for harvard_oxford or aal for aal
    '''
    
    if atlas == 'ho':
        return nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    if atlas == 'aal':
        return nilearn.datasets.fetch_atlas_aal()
    
def labels(coords: tuple, atlas_img: str, labels: list, atlas='ho', aal_index=False ) -> pd.DataFrame:

    '''
    Function to get brain region based on atlases

    Parameters
    ----------
    coords: list,
        list of coords
    atlas_img: str  
        str to atlas atlas image
    labels: list
        list to

    Returns
    -------
    pd.Series: Series 
        Series of labels
        
    '''
    
    get_labels = NiftiSpheresMasker((coords.values)).fit_transform(atlas_img).ravel()
    if atlas == 'ho':
        return pd.Series([labels[int(ind)] for ind in np.nditer(get_labels)])
    if atlas == 'aal':
        aal_labels = []
        for ind in np.nditer(get_labels):
            try:
                aal_labels.append(labels[aal_index.index(str(int(ind)))])
            except Exception:
                aal_labels.append('no label')
        return pd.Series(aal_labels)

def left_or_right(x) -> str:

    '''
    Function to decide if brain region is left or right
    depending on x co-ordinate

    Parameters
    ----------
    x: float
        x co-ordinate
    
    Returns
    -------
    str: string
        str of left or right
    
    '''
    if x < 0:
        return 'Left '
    else:
        return 'Right '

def predictors_df(df, coords) -> pd.DataFrame:

    '''
    Function to get labels for predictors df

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of x, y, z mni co-ordinates
        and betas
    coords: list
        list of co-ordinates
    
    Returns
    -------
    df: pd.DataFrame
        DataFrame of predictors with labels organised 
        by beta weights
    '''
    havard_atlas = fetch_atlas('ho')
    aal_atlas = fetch_atlas('aal')
    havard_labels = labels(coords, havard_atlas['filename'], havard_atlas['labels'])
    aal_labels = labels(coords, 
                             aal_atlas['maps'], 
                             aal_atlas['labels'], 
                             atlas='aal', 
                             aal_index=aal_atlas['indices'])
    df['label'] = havard_labels
    df['label'] = df.apply(lambda x : left_or_right(x.X) + x.label, axis=1)
    df['label_aal'] = aal_labels    
    df['label'] = df['label'].apply(lambda val: np.nan if 'Background' in val else val)
    df['labels'] = df['label'].fillna(df['label_aal'])
    return df.drop(['label', 'label_aal'], axis=1).sort_values(by=['beta'])

def load_mvpa_predictors(csv_name: str) -> pd.DataFrame:

    '''
    Function to load MVPA predictors. Returns csv
  

    Parameters
    ---------
    path: str
        absolute path to csv

    Returns
    -------
    pd.DataFrame of cluster csv
    '''
    df = pd.read_csv(csv_name)
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Cluster Size (mm3)', 'beta', 'pval', 'BFB', 'odds', 'null_proability' ,'labels']
    df =  bayesian_cluster_info(df)
    return df[cols]
