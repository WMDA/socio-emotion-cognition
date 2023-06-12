import pandas as pd
from decouple import config
import glob 
import os
from itertools import chain
import re
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

def get_rt(experimental_dfs: list) -> dict:

    '''
    Function to return RT in list format.

    Parameters
    ----------
    experimental_dfs: list 
        list of paths to dataframes

    Returns
    -------
    rt_values: dict 
        lists of rt values and subjects
    '''
    
    
    values_dictionary = {
        'rt_values': [],
        'subject': []
    }
    for dataframes in experimental_dfs:
        df = pd.read_csv(dataframes, sep='\t')
        if len(df.columns) < 3:
            df = pd.read_csv(dataframes)
        column = 'RT'
        df['subject'] = re.findall(r'\D\d\d..', dataframes)[0]
        values_dictionary['subject'].append(df['subject'])
        
        if column not in df.columns:
            column = 'response_time' 
        if df[column].dtype != 'int64':
            df[column] = df[column].str.replace('.', '0').astype(int)
        values_dictionary['rt_values'].append(df[column])
    return {'values': list(chain.from_iterable(values_dictionary['rt_values'])),
            'subjects': values_dictionary['subject']
            }

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
        reaction_times[key].append(values['values'])
        reaction_times[key].append(values['subjects'])
   
    return reaction_times

def rt_df(rts: dict, task) -> pd.DataFrame:
    
    '''
    Function to put reaction times into long form df
    with group, time and rt columns

    Parameters
    ----------
    rts: dict
        dictionary of reaction times by group and time
    task: str
        string object to dictate task being used. Needed
        to calculate subjects correctly
    
    Returns
    -------
    pd.DataFrame: dataframe object
        Dataframe of group, time point, subjects and rt values
    '''
    
    length = 73
    if task == 'eft':
        length = 39

    rt_column = rts['HC_t1'][0][0] + rts['HC_t2'][0][0] + rts['AN_t1'][0][0] + rts['AN_t2'][0][0]
    group_column = ['HC' for group in range(len(rts['HC_t1'][0] + rts['HC_t2'][0]))] + ['AN' for group in range(len(rts['AN_t1'][0] + rts['AN_t2'][0]))] 
    time_column = ['t1' for time in rts['HC_t1'][0]] + ['t2' for time in rts['HC_t2'][0]] + ['t1' for time in rts['AN_t1'][0]] + ['t2' for time in rts['AN_t2'][0]]
    subject_column = list(chain.from_iterable(rts['HC_t1'][1])) + list(chain.from_iterable(rts['AN_t1'][1])) + list(chain.from_iterable(rts['HC_t1'][1])) + list(chain.from_iterable(rts['AN_t1'][1])) + ['G9999' for no in range(length)]
    return pd.DataFrame({'rt': rt_column, 'group': group_column,'time': time_column,'subject': subject_column })

def organise_df(values: list, task: str, which_task: bool):

    '''
    Function to decide to process which

    Parameters
    ----------
    values: list
        list of values for either rt or is the task correct
    
    task: str
        string object to dictate task being used. Needed
        to calculate subjects correctly
    
    which _task: bool
       If False assumes values are rt if True assumes 
       values are is task correct.

    Returns
    -------
    pd.DataFrame: dataframe object
        Dataframe of group, time point and rt values
    '''
    if which_task == False:
        return rt_df(values, task)


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
    df = organise_df(rts, task, task_correct)
    return df

def get_mean(path: str, condition:str) -> float:

    '''
    Function to get the mean.

    Parameters
    ----------
    path: str
        str of path to dataframe
    condition: str
        str of condition to filter df
    
    Returns
    -------
    float of mean
    '''


    df = pd.read_csv(path, sep='\t')
    if len(df.columns) < 3:
            df = pd.read_csv(path)
    try:
        df = df[df['Condition']== condition]
    except Exception:
        df = df[df['trial_type']== condition]
    
    column = 'RT'
    if column not in df.columns:
        column = 'response_time' 
    if df[column].dtype != 'int64':
        df[column] = df[column].str.replace('.', '0').astype(int)

    return df[column].mean()

def get_rt_summary(experimental_dfs: pd.DataFrame, condition: str) -> pd.DataFrame:

    '''
    Function to return RT in dataf format.

    Parameters
    ----------
    experimental_dfs: list 
        list of paths to dataframes
    condition: str
        str of condition to filter df

    Returns
    -------
    pd.DataFrame: DataFrame
        pd.DataFrame of rts with group
        time, rt value and subject columns
    '''
    
    
    values_dictionary = {
        'rt_values': [],
        'subject': [],
        'time_point':[],
        'group':[],
    }
    
    subject = 1
    for row in experimental_dfs.iterrows(): 
        try:
            values_dictionary['rt_values'].append(get_mean(row[1]['t1'], condition))
            values_dictionary['rt_values'].append(get_mean(row[1]['t2'], condition))
            values_dictionary['time_point'].append('t1')
            values_dictionary['time_point'].append('t2')
            values_dictionary['subject'].append(subject)
            values_dictionary['subject'].append(subject)
            subject += 1
    
            group = 'HC'
            if 'sub-G2' in row[1]['t1']:
                group = 'AN'
            values_dictionary['group'].append(group)
            values_dictionary['group'].append(group)
        except Exception as e:
            print(e)
            continue
            
    return pd.DataFrame({'rt': values_dictionary['rt_values'],
                         'group': values_dictionary['group'],
                         'time': values_dictionary['time_point'],
                         'sub': values_dictionary['subject'],
                         })

def get_subjects(task) -> pd.DataFrame:

    '''
    Function to get subjects in order.
    
    Parameters
    ----------
    task: str
        str of task name
    
    Returns
    -------
    df: pd.DataFrame
        Dataframe with two columns
        t1 and t2 with participants in order
    '''
    df_location = config(task)
    df = pd.read_csv(os.path.join(df_location, '1stlevel_location.csv'))
    for col in df.columns:
        df[col] = df[col].str.findall(r'sub-.....')
    return df

def get_dfs(df: pd.DataFrame, task: str) -> pd.DataFrame:
    
    '''
    Function to get paths to experimental dfs

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of subjects in order
    task: str
        str of task

    Returns
    -------
    df: pd.DataFrame
       Dataframe of file paths in two columns 
    '''    
    
    experiment_dir = config('raw_data')
    df['t1'] = df['t1'].apply(lambda particpant: f'{experiment_dir}/bids_t1/{particpant[0]}/func/{particpant[0]}_task-{task}_events.tsv')
    df['t2'] = df['t2'].apply(lambda particpant: f'{experiment_dir}/bids_t2/{particpant[0]}/func/{particpant[0]}_task-{task}_events.tsv')
    return df


def get_mean_rt_df(task: str, condition:str) -> pd.DataFrame:
    
    '''
    Function wrapper to get the mean rt of participants
    in long form df

    Parameters
    ----------
    task: str
        str of task
    condition: str
        str of condition to filter df

    Returns
    -------
    pd.DataFrame of rts with group
    time, rt value and subject columns
    
    '''
    subjects = get_subjects(task)
    df = get_dfs(subjects, task)
    df['t2'] = df['t2'].str.replace('B2024', 'B2024B')
    return get_rt_summary(df, condition)