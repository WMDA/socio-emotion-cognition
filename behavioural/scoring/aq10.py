from fNeuro.behavioural.data_functions import load_data
from fNeuro.behavioural.scoring_functions import aq10_dict, imputate, score
import pandas as pd
import warnings
import re
# To ignore all pandas .loc slicing suggestions
warnings.filterwarnings(action='ignore')


def aq10_scoring() -> pd.DataFrame:

    '''
    Main function for AQ10 scoring

    Parameters
    ----------
    None

    Returns
    -------
    aq_score: pd.DataFrame Scores for AQ10. 
    '''
    

    df = load_data('BEACON', table='raw_t2_all_values')
    
    df = df[df['q7'].str.contains(r'_2|_3', regex=True)]
    df = df.drop(df[df['q7'].str.contains('B2064', regex=False)].index[0])
    df['q7'] = df['q7'].apply(lambda part: re.sub(r'_.', '', part))
    scores = aq10_dict()
    
    aq_df = df.loc[:, 'q87':'q96']
    aq_df['7.'] = df['q7']
    
    agree_df = aq_df[['7.','q87', 'q93', 'q94', 'q96']]
    disagree_df = aq_df[['7.','q88', 'q89', 'q90', 'q91', 'q92', 'q95']]
    
    for column in agree_df.columns:
        if column != '7.':
            agree_df[column] = agree_df[column].apply(lambda value: scores['agree'][value] if type(value) == str else value )
    
    for column in disagree_df.columns:
        if column != '7.':
            disagree_df[column] = disagree_df[column].apply(lambda value: scores['disagree'][value] if type(value) == str else value)
    
    aq_score = pd.concat([agree_df, disagree_df.drop('7.', axis=1)], axis=1).rename(columns={'7.': 'q7'})
    aq_score['group'] = aq_score['q7'].apply(lambda x: 'HC' if 'B1' in x else 'AN')
    aq_score = score(aq_score)
    return aq_score[['B_Number', 'overall_score', 'group']]


if __name__ == '__main__':
    aq10 = aq10_scoring()
    print(aq10)