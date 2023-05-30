import pandas as pd
from decouple import config
import glob 
import os
from itertools import chain
import warnings
warnings.filterwarnings("ignore")

def get_experimental_dataframes(task) -> dict:

    experiment_dir = config('raw_data')
    hc_t1 = glob.glob(os.path.join(experiment_dir, 'bids_1','sub-G1*', 'func', f'*_task-{task}_events.tsv'))
    an_t1 = glob.glob(os.path.join(experiment_dir, 'bids_1','sub-G2*', 'func', f'*_task-{task}_events.tsv'))
    hc_t2 = glob.glob(os.path.join(experiment_dir, 'bids_2','sub-B1*', 'func', f'*_task-{task}_events.tsv'))
    an_t2 = glob.glob(os.path.join(experiment_dir, 'bids_2','sub-B2*', 'func', f'*_task-{task}_events.tsv'))
    return {
        'AN_t1': an_t1,
        'HC_t1': hc_t1,
        'AN_t2': an_t2,
        'HC_t2': hc_t2
    }

def get_rt(experimental_dfs) -> list:
    
    '''
    Function to return RT in list format.

    Parameters
    ----------
    experimental_dfs: list of paths to dataframes

    Returns
    -------
    rt_values: list of rt values unordered
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
    return list(chain.from_iterable((rt_values)))

def reaction_times(task: str) -> pd.DataFrame:  
    df_paths = get_experimental_dataframes(task)
    hc_values = get_rt(df_paths['HC_t1'])
    an_values = get_rt(df_paths['AN'])
    return pd.DataFrame(data={
                              'HC': hc_values, 
                              'AN': an_values
                              })


df = reaction_times('fear')
print(df.shape)