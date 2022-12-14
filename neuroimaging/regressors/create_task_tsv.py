import argparse
import os
import pandas as pd

def options() -> dict:
    flags = argparse.ArgumentParser()
    flags.add_argument('-d', '--dir', dest='dir', help='directory where task files are saved')
    flags.add_argument('-s', '--save', dest='save', help='BIDS directory to save tsvs to') 
    flags.add_argument('--stim', dest='stim', help='Defines stimulus name i.e happy of fear' )
    arg = vars(flags.parse_args())
    return arg

def get_files(directory:str) -> list:
	files = os.listdir(directory)
	return files

def duration(df:pd.DataFrame) -> pd.Series:
    duration =  df['TimeAtStartOfTrial'].shift(periods= -1) - df['TimeAtStartOfTrial']
    duration = duration.fillna(duration.mean()).rename('duration')
    return duration

def stimuli(col:str, stim:str) -> str:
    if '4' in col:
        return f'{stim}'
    elif '2' in col:
        return f'Partially_{stim}'
    elif '0' in col:
        return 'Neutral'
    else:
        return 'Blank'


def create_tsv(csv:str, stim:str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    dur = duration(df)
    onset = df['TimeAtStartOfTrial'].rename('onset')
    trial_type = df['ImageFile'].apply(lambda col: stimuli(col, stim))
    stim_file = df['ImageFile'].rename('stim_file')
    response_time = df['RT'].rename('response_time')  
    tsv = pd.concat([onset, dur, trial_type.rename('trial_type'), response_time, stim_file], axis=1)
    return tsv

if __name__ == '__main__':
    flags = options()
    files = get_files(flags['dir'])
    

    for file in files:
        csv = flags['dir'] + file
        
        try:
            tsv = create_tsv(csv, flags['stim'])
            print(tsv)
        except Exception as e:
            print(e)
        

