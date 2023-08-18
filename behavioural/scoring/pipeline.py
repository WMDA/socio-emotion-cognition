import pandas as pd
from edeq import edeq_scoring
from hads import hads_scoring
from time_difference import time_diff
from fNeuro.behavioural.data_functions import connect_to_database

def bmi_calculation() -> pd.DataFrame:
    
    '''
    Gets the height, weight and BMI
    of participants 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    bmi_df: pd.DataFrame
        df containing bmi, cm and height
    '''
    
    bmi_df = pd.read_csv("weight.csv").dropna()
    bmi_df['weight'] = bmi_df['weight'].astype('float64')
    bmi_df['bmi'] = (bmi_df['weight'] / (bmi_df['cm'] **2)) * 10000
    return bmi_df

if __name__ == '__main__':
    connector = connect_to_database('BEACON')
    measures = {
        'hads_post_break': hads_scoring(),
        'edeq_post_break': edeq_scoring(),
        'time_post_break': time_diff(),
        'bmi_neuroimaging': bmi_calculation(),
        'neuroimaging_index': pd.read_csv('index.csv').drop('participant', axis=1)
}

    df = measures['edeq_post_break']
    for key in measures.keys():
        measures[key].to_sql(key, connector) 