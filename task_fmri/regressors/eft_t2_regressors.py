import pandas as pd
from decouple import config
import os


def duration(df: pd.DataFrame) -> pd.Series:
    '''
    Function to calculate duration of trial

    Parameters
    ----------
    df: pd.Dataframe of trials

    Returns
    -------
    duration: pd.Series of duration of trial

    '''
    duration = df['TimeAtStartOfTrial'].shift(
        periods=-1) - df['TimeAtStartOfTrial']
    duration = duration.fillna(duration.mean()).rename('duration')
    return duration


def create_tsv(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function to create tsv file for ebedded figures task

    Parameters
    ---------
    df: pd.DataFrame object
        Dataframe of raw task behaviour

    Returns
    -------
    pd.Dataframe with calculated onset, 
        duration, Condition, Filename, CorrectSide, 
        ChosenSide, IsCorrect', RT
    '''

    durations = duration(df)  
    onset = df['TimeAtStartOfTrial'].rename('onset')
    return pd.concat([onset, durations, df[['Condition', 'Filename', 'CorrectSide', 'ChosenSide', 'IsCorrect', 'RT']]], axis=1)


if __name__ == '__main__':
    
    base_dir = config('eft_t2_task_files')
    bids_t2_dir = config('bids_t2')
    files = os.listdir(base_dir)
    
    for file in files:
        subject = file.split('_')[2]
        if 'B2024' in subject:  # An exception in naming occurs for one participant
            subject: str = 'B2024B'
        print(f'Analysing subject sub-{subject}')
        df = pd.read_csv(f'{base_dir}{file}')
        tsv_file = create_tsv(df)
        print(f'{bids_t2_dir}sub-{subject}/func/sub-{subject}_task-eft_events.tsv')
        tsv_file.to_csv(f'{bids_t2_dir}sub-{subject}/func/sub-{subject}_task-eft_events.tsv')