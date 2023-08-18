from fNeuro.behavioural.data_functions import load_data
from fNeuro.behavioural.scoring_functions import edeq, score, imputate
import pandas as pd
import re
import warnings
# To ignore all pandas .loc slicing suggestions
warnings.filterwarnings(action='ignore')


def edeq_scoring() -> pd.DataFrame:
   
    '''
    Main function for EDEQ scoring

    Parameters
    ----------
    None

    Returns
    -------
    edeq_scores: pd.DataFrame: Scores for ede-q. 
    
    '''

    df = load_data('BEACON', table='raw_t2_all_values')
    
    df = df[df['q7'].str.contains(r'_2|_3', regex=True)]
    df = df.drop(df[df['q7'].str.contains('B2064', regex=False)].index[0])
    df['q7'] = df['q7'].apply(lambda part: re.sub(r'_.', '', part))
    
    edeq_df = df[['q7', 'q25', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31', 'q33', 'q34', 'q35', 'q36',
                      'q37', 'q38', 'q39', 'q47', 'q48', 'q49', 'q50', 'q51', 'q52', 'q53', 'q54']]
    
    edeq_df = edeq(edeq_df)
    edeq_df = imputate(edeq_df)
    restraint = edeq_df[['q7','group', 'q25', 'q26', 'q27', 'q28', 'q29']]
    eating_concern = edeq_df[['q7','group', 'q30', 'q31', 'q33', 'q52', 'q39']]
    shape_concern = edeq_df[['q7','group','q34', 'q35', 'q48',
                         'q36', 'q51', 'q53', 'q54', 'q37']]
    weight_concern = edeq_df[['q7','group','q47', 'q49', 'q35', 'q50', 'q38']]
    restraint_score = score(restraint)
    restraint_score['restraint'] = restraint_score['overall_score'] / 5
    
    eating_concern_score = score(eating_concern)
    eating_concern_score['eating_concern'] = eating_concern_score['overall_score'] / 5
    shape_concern_score = score(shape_concern)
    shape_concern_score['shape_concern'] = shape_concern_score['overall_score'] / 8
    
    weight_concern_score = score(weight_concern)
    weight_concern_score['weight_concern'] = weight_concern_score['overall_score'] / 5
    
    edeq_scores = pd.concat([restraint_score[['B_Number','restraint']], eating_concern_score['eating_concern'],
                             shape_concern_score['shape_concern'], weight_concern_score['weight_concern']], axis=1)
    edeq_scores['global_score'] = edeq_scores.drop('B_Number', axis=1).sum(axis=1)/4
    edeq_scores['group'] = weight_concern_score['group']

    return edeq_scores


if __name__ == '__main__':
    edeq_score = edeq_scoring()
    print(edeq_score)
    
