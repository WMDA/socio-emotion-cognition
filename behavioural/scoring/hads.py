from fNeuro.behavioural.data_functions import load_data
from fNeuro.behavioural.scoring_functions import clean_up_columns, imputate, score
import pandas as pd
import re

def hads_scoring(verbose=False):
    '''
    Main function for scoring hads. 

    Returns
    ------- 
    hads_scores: pandas dataframe of hads results
    '''

    df = load_data('BEACON', table='raw_t2_all_values')
    df = df[df['q7'].str.contains(r'_2|_3', regex=True)]
    df = df.drop(df[df['q7'].str.contains('B2064', regex=False)].index[0])
    df['q7'] = df['q7'].apply(lambda part: re.sub(r'_.', '', part))
    hads_df = df.loc[:, 'q73':'q86']
    hads_df['q7'] = df['q7']

    hads_df = clean_up_columns(hads_df)
    hads_df = imputate(hads_df)

    anxiety = score(hads_df[['q7','q73', 'q75', 'q77',
                             'q79', 'q81', 'q83', 'q85', 'group']]).rename(columns={'overall_score': 'anxiety'})
    depression = score(hads_df[['q7', 'q74', 'q76', 'q78',
                                'q80', 'q82', 'q84', 'q86', 'group']])

    hads_score = pd.concat([anxiety, depression.drop(['B_Number', 'group'], axis=1)], axis=1).rename(columns={'overall_score': 'depression'})

    hads_score = hads_score[['B_Number', 'anxiety', 'depression', 'group' ]]
    
    return hads_score


if __name__ == '__main__':
    hads = hads_scoring(verbose=True)
    print(hads)