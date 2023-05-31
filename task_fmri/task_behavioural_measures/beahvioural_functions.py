import pandas as pd
from decouple import config
import glob 
import os
from itertools import chain
import warnings
warnings.filterwarnings("ignore")

def get_experimental_dataframes(task: str) -> dict:
    
    '''
    Function to get all the event files by group 
    and time point

    Parameters
    ----------
    task: str
        str of task

    Returns
    -------
    dict: dictionary object
        dict of paths for event files
    '''
    
    experiment_dir = config('raw_data')
    hc_t1 = glob.glob(os.path.join(experiment_dir, 'bids_t1','sub-G1*', 'func', f'*_task-{task}_events.tsv'))
    an_t1 = glob.glob(os.path.join(experiment_dir, 'bids_t1','sub-G2*', 'func', f'*_task-{task}_events.tsv'))
    hc_t2 = glob.glob(os.path.join(experiment_dir, 'bids_t2','sub-B1*', 'func', f'*_task-{task}_events.tsv'))
    an_t2 = glob.glob(os.path.join(experiment_dir, 'bids_t2','sub-B2*', 'func', f'*_task-{task}_events.tsv'))
    return {
        'AN_t1': an_t1,
        'HC_t1': hc_t1,
        'AN_t2': an_t2,
        'HC_t2': hc_t2
    }

def get_rt(experimental_dfs: list) -> list:
    
    '''
    Function to return RT in list format.

    Parameters
    ----------
    experimental_dfs: list 
        list of paths to dataframes

    Returns
    -------
    rt_values: list 
        list of rt values unordered
    '''
    
    rt_values = []
    for dataframes in experimental_dfs:
        df = pd.read_csv(dataframes, sep='\t')
        if len(df.columns) < 3:
            df = pd.read_csv(dataframes)
        column = 'RT'
        if column not in df.columns:
            column = 'response_time' 
        if df[column].dtype != 'int64':
            df[column] = df[column].str.replace('.', '0').astype(int)
        rt_values.append(df[column])
    return list(chain.from_iterable(rt_values))

def iscorrect(experimental_dfs: list) -> list:
    
    '''
    Function to return RT in list format.

    Parameters
    ----------
    experimental_dfs: list 
        list of paths to dataframes

    Returns
    -------
    rt_values: list 
        list of rt values unordered
    '''
    
    iscorrect = []
    for dataframes in experimental_dfs:
        df = pd.read_csv(dataframes, sep='\t')
        if len(df.columns) < 3:
            df = pd.read_csv(dataframes)
        column = 'RT'
        if column not in df.columns:
            column = 'response_time' 
        if df[column].dtype != 'int64':
            df[column] = df[column].str.replace('.', '0').astype(int)
        rt_values.append(df[column])
    return list(chain.from_iterable(rt_values))


def loop_groups(df_paths: list, task_correct=False) -> dict:  

    '''
    Function tp loop through group and time points
    obtaining outcome.

    Parameters
    ----------
    df_paths: list
        list of paths to events df
    task_correct: bool
        set to true to analyse if task was correct 
        default is reaction time
    
    Returns
    -------
    dict: dictionary object
        dict of reaction times by group and 
        time point
    '''

    function_to_eval = get_rt

    reaction_times = dict(zip([key for key in df_paths.keys()], 
                              [list() for key in range(len(df_paths))]))
    
    for key in df_paths.keys():
        values = function_to_eval(df_paths[key])
        reaction_times[key].append(values)
    return reaction_times

def rt_df(rts: dict) -> pd.DataFrame:
    
    '''
    Function to put reaction times into long form df
    with group, time and rt columns

    Parameters
    ----------
    rts: dict
        dictionary of reaction times by group and time
    
    Returns
    -------
    pd.DataFrame: dataframe object
        Dataframe of group, time point and rt values
    '''
    rt_column = rts['HC_t1'][0] + rts['HC_t2'][0] + rts['AN_t1'][0] + rts['AN_t2'][0]
    group_column = ['HC' for group in range(len(rts['HC_t1'][0] + rts['HC_t2'][0]))] + ['AN' for group in range(len(rts['AN_t1'][0] + rts['AN_t2'][0]))] 
    time_column = ['t1' for time in range(len(rts['HC_t1'][0]))] + ['t2' for time in range(len(rts['HC_t2'][0]))] + ['t1' for time in range(len(rts['AN_t1'][0]))] + ['t2' for time in range(len(rts['AN_t2'][0]))]
    return pd.DataFrame({'rt': rt_column, 'group': group_column, 'time': time_column})




def organise_df(values: list, which_task: bool):

    '''
    Function to decide to process which

    Parameters
    ----------
    values: list
        list of values for either rt or is the task correct
    
    which _task: bool
       If False assumes values are rt if True assumes 
       values are is task correct.

    Returns
    -------
    pd.DataFrame: dataframe object
        Dataframe of group, time point and rt values
    '''
    if which_task == False:
        return rt_df(values)


def long_df(task: str, task_correct: bool=False) -> pd.DataFrame:
    
    '''
    Function wrapper to return long_df of values.
    Will accept either is task correct or rt.

    Parameters
    ----------
    task: str
        str of name of task, either fear, happy or eft
    task_correct: bool default False
        If set to true will do is task correct rather
        than rt.
    
    Returns
    -------
    pd.DataFrame: dataframe object
        Dataframe of group, time point and rt values
    '''
    
    df_paths = get_experimental_dataframes(task)
    rts = loop_groups(df_paths)
    df = organise_df(rts, task_correct)
    return df