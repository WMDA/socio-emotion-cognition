from fNeuro.behavioural.data_functions import load_data
import pandas as pd
import numpy as np
import re
import warnings
# To ignore all pandas .loc slicing suggestions
warnings.filterwarnings(action='ignore')


def time_diff():
    '''
    Main function to calculate length of time between time point and time point 2

    Parameters
    ----------
    None

    Returns
    -------
    time_df:pandas df: Dataframe with difference in days and years between time point one and time point two.
    '''

    df_t2 = load_data('BEACON', 'raw_t2_all_values')
    df_t1 = load_data('BEACON', 'participant_index')
    df_t1['t2'].iloc[df_t1[df_t1['t2'].str.contains('B2091', regex=False)].index] = 'B2999'
    df_t1['initial'].iloc[df_t1[df_t1['t1'].str.contains('G2142', regex=False)].index] = '22/07/2021'
    df_t2 = df_t2[df_t2['q7'].str.contains(r'_2|_3', regex=True)].reset_index(drop=True)
    df_t2['q7'] = df_t2['q7'].apply(lambda part: re.sub('_2', '', part))
    df_t2 = df_t2.drop(df_t2[df_t2['q7'].str.contains('B2064', regex=False)].index[0])
    df_t2['q7'] = df_t2['q7'].apply(lambda part: re.sub('_3', '', part))
    df_t2['q7'] = df_t2['q7'].str.rstrip()
    group = df_t2[['q7', 'time_finished']]
    df_t1 = pd.merge(df_t1, df_t2['q7'], left_on='t2', right_on='q7').drop('q7', axis=1)
    hc = group[group['q7'].str.contains('B1')]
    an = group[group['q7'].str.contains('B2')]
    hc['group'] = 'HC'
    an['group'] = 'AN'

    time_t2 = pd.concat([hc, an])
    time_t2.sort_values(by=['q7'], inplace=True)
    time_t2 = time_t2.reset_index(drop=True)

    time_points = pd.concat([df_t1, time_t2['time_finished']], axis=1)
    time_points['finished'] = time_points['time_finished'].apply(
        lambda value: re.sub(r'..:..:.. UTC', '', value) if type(value) == str else value)
    t1_dates = pd.to_datetime(time_points['initial'], dayfirst=True, utc=True)
    t2_dates = pd.to_datetime(time_points['time_finished'], utc=True)


    difference = pd.DataFrame()
    difference['days'] = t2_dates - t1_dates
    difference['years'] = difference['days'] / np.timedelta64(1, 'Y')

    time_df = pd.concat([time_points[['t1', 't2']], difference[[
                        'days', 'years']], time_t2['group']], axis=1)
    
    return time_df

if __name__ == '__main__':
    time_df = time_diff()
    print(time_df)