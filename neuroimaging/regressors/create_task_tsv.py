import argparse
import os
import pandas as pd


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
    flags.add_argument('-s', '--save', dest='save',
                       help='BIDS directory to save tsvs to')
    flags.add_argument('--stim', dest='stim',
                       help='Defines stimulus name i.e happy of fear')
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
    returns str of stimulus name
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


if __name__ == '__main__':
    flags: dict = options()
    files: list = os.listdir(flags['dir'])

    for file in files:
        csv_location: str = flags['dir'] + file

        try:
            tsv: pd.DataFrame = create_tsv(csv_location, flags['stim'])

            print(csv_location.rstrip('.csv'))
        except Exception as e:
            print(e)
