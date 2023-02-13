import argparse
import os
import pandas as pd
import re
from decouple import config
import sys

def options() -> dict:
    '''
    Function to accept accept command line flags

    Parameters
    ---------
    None

    Returns
    -------
    dictionary of flags given
    '''
    flags = argparse.ArgumentParser()
    flags.add_argument('-d', '--dir', dest='dir',
                       help='directory where task files are saved')
    flags.add_argument('--stim', dest='stim',
                       help='Defines stimulus name i.e happy of fear')
    flags.add_argument('--time', dest='time',
                       help='Defines time point to get data from')
    flags.add_argument('--subject', dest='subject',
                       help='File path for individual subject instead of a group of individuals')
    return vars(flags.parse_args())


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


def stimuli(stimuls_shown: str, stim: str) -> str:
    '''
    Function to calculate name of stimulus seen

    Parameters
    ----------
    stimuls_shown: str of stimulus shown
    stim: str of if this happy or sad stimulus

    Returns
    -------
    str of stimulus name
    '''
    if '4' in stimuls_shown:
        return f'{stim}'
    elif '2' in stimuls_shown:
        return f'Partially_{stim}'
    elif '0' in stimuls_shown:
        return 'Neutral'
    else:
        return 'Blank'


def create_tsv(csv: str, stim: str) -> pd.DataFrame:
    '''
    Wrapper function to read in csv file then calculate duration,
    onset, trial type and stim file

    Parameters
    ----------
    csv: str name of csv file
    stim: str of stimulus type being shown

    Returns
    -------
    tsv: pd.Dataframe of BIDS compliant dataframe
    '''
    df: pd.DataFrame = pd.read_csv(csv)
    dur: pd.Series = duration(df)
    onset: pd.Series = df['TimeAtStartOfTrial'].rename('onset')
    trial_type: pd.Series = df['ImageFile'].apply(
        lambda col: stimuli(col, stim))
    stim_file: pd.Series = df['ImageFile'].rename('stim_file')
    response_time: pd.Series = df['RT'].rename('response_time')
    tsv: pd.DataFrame = pd.concat([onset, dur, trial_type.rename(
        'trial_type'), response_time, stim_file], axis=1)
    return tsv


def path_to_data(time_point: str) -> str:
    '''
    Function to get path to raw data

    Parameters
    ----------
    time_point: str of time point number

    Returns
    -------
    str of path to bids_directory
    '''
    raw_data: str = config('raw_data')
    return os.path.join(raw_data, f'bids_t{time_point}')


def file_path(time_point: str, csv_location: str) -> dict:
    '''
    Function to get file path to save data to

    Parameters
    ----------
    time_point: str of which time point to save data to
    csv_location: str of csv file location

    Returns
    -------
    file_info: dict of file path and subject info.
    '''
    bids_directory: str = path_to_data(time_point)
    number: str = ''.join(re.findall(r"[.\d*?]\d", csv_location))
    if '1' in time_point:
        prefix: str = 'G'
    if '2' in time_point:
        prefix: str = 'B'

    subject: str = 'sub-' + prefix + number

    if 'B2024' in subject:  # An exception in naming occurs for one participant
        subject: str = 'sub-B2024B'

    file_info = {
        'subject': subject,
        'path': f'{bids_directory}/{subject}/func'
    }

    return file_info


def analyse_subject(flags: dict, file: str ='None') -> None:
    '''
    Function to analyse subjects. Saves tsv file to correct
    bids directory.

    Parameters
    ----------
    flags: dict of arguments.
    file:  str of file name. Used with -d/--directory flag 
           otherwise set None.

    Returns
    -------
    None
    '''
    if '1' in flags['time']:
        suffix: str = 'sub-G'
    else:
        suffix: str = 'sub-B' 

    if flags['subject'] != None:
        csv_location: str = flags['subject'] 
        subject: str = suffix + os.path.basename(flags['subject']).split('_')[2]
    else:
        csv_location: str = flags['dir'] + file
        subject: str = suffix + file.split('_')[2]
    
    print(f"\nAnalysing subject {subject}\n")
    
    try:  
        bids_file_path: dict = file_path(flags['time'], csv_location)
        tsv: pd.DataFrame = create_tsv(csv_location, flags['stim'])
        tsv.to_csv(
                    f"{bids_file_path['path']}/{bids_file_path['subject']}_task-{flags['stim']}_events.tsv"
                  )

    except Exception as e:
        print(f'\nUnable to analyse {subject} due to {e}\n')


if __name__ == '__main__':
    flags: dict = options()

    if flags['subject'] != None:
        analyse_subject(flags)
        print('Finished')
        sys.exit(0)
    
    files: list = os.listdir(flags['dir'])
    for file in files:
        analyse_subject(flags, file=file)
    print('Finished')
    sys.exit(0)