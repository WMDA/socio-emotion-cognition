import pandas as pd
from fNeuro.behavioural.data_functions import load_data, connect_to_database


if __name__ == '__main__':
    
    # First load in all the measures
    measures = {
        'edeq_t1': load_data('BEACON', 'edeq_t1' ),
        'edeq_t2': load_data('BEACON', 'edeq_t2' ),
        'edeq_post_break': load_data('BEACON', 'edeq_post_break'),
        'hads_t1': load_data('BEACON', 'hads_t1'),
        'hads_t2': load_data('BEACON', 'hads_t2'),
        'hads_post_break': load_data('BEACON', 'hads_post_break'),
        'time_difference': load_data('BEACON', 'time_difference'),
        'time_post_break': load_data('BEACON', 'time_post_break'),
        'bmi_neuroimaging_t2': load_data('BEACON', 'bmi_neuroimaging'),
        'bmi_t1': load_data('BEACON', 'bmi_t1'),
        'participant_index': load_data('BEACON', 'neuroimaging_index'),
        'age': load_data('BEACON','raw_t1')
        }
    
    # Some of the participants have trailing white space
    measures['participant_index']['t2'] = measures['participant_index']['t2'].str.rstrip()
    part_index = measures['participant_index'].sort_values(by='t2')

    # Merge onto index to remove participants who didn't take part then merge t1 and t2 dfs together
    ede_df = pd.merge(left=pd.merge(right=measures['edeq_t2'].rename(columns={'global_score': 'gs_t2'}), 
                                    left=part_index, 
                                    left_on='t2', 
                                    right_on='B_Number', 
                                    how='left'),
                      right=measures['edeq_t1'].rename(columns={'Total Score': 'gs_t1'}), 
                      left_on='t1', 
                      right_on='G_Number', 
                      how='left').drop(['index_x', 'index_y', 'group_x',  'group_y',
                                        'eating_concern', 'shape_concern', 'weight_concern',
                                        'Restraint', 'Eating Concern', 'Shape Concern','Weight Concern', 
                                        'B_Number', 'restraint', 'index', 'G_Number', ], axis=1)
                      
    # Get just T2 participants
    edeq_t2 = pd.merge(right=measures['edeq_t2'].rename(columns={'global_score': 'gs_t2'}), 
                                    left=part_index, 
                                    left_on='t2', 
                                    right_on='B_Number', 
                                    how='left').drop(['index_x', 'eating_concern', 'shape_concern', 
                                                      'weight_concern', 'B_Number', 'restraint', 'index_y', 'group', 't1'],  axis=1)
    # Drop all the participants with repeated measures 
    replace_ede_q = edeq_t2[~edeq_t2['t2'].isin(measures['edeq_post_break']['B_Number'])].reset_index(drop=True)
    replace_ede_q = replace_ede_q[replace_ede_q['t2'] != 'B2024b'] # This participant needs to be dropped manually
    
    # Get the repeated measures & rename columns and strip trailing white space 
    repeated_measure = measures['edeq_post_break'][['B_Number', 'global_score']].rename(columns={'B_Number': 't2', 
                                                                                        'global_score': 'gs_t2'})
    repeated_measure['t2'] = repeated_measure['t2'].str.rstrip()
    replace_ede_q['t2'] = replace_ede_q['t2'].str.rstrip()
    
    # Merge the dataframes and concat with t1 data 
    t2_ede = pd.merge(replace_ede_q, repeated_measure, how='outer').sort_values(by='t2').reset_index(drop=True)
    edeq = pd.concat((ede_df[['t1', 'gs_t1']], t2_ede), axis=1) 
    
    # Hads does exactly the same as the ede-q. Could not write into function due to renaming of multiple columns
    hads_df =  pd.merge(left=pd.merge(right=measures['hads_t2'].rename(columns={'anxiety': 'anxiety_t2', 'depression': 'depression_t2'}), 
                                    left=part_index, 
                                    left_on='t2', 
                                    right_on='B_Number', 
                                    how='left'),
                      right=measures['hads_t1'].rename(columns={'anxiety': 'anxiety_t1', 'depression': 'depression_t1'}), 
                      left_on='t1', 
                      right_on='G_Number', 
                      how='left').drop(['index_x', 'index_y', 'group_x',  'group_y',
                                        'B_Number', 'index', 'G_Number', ], axis=1)
    hads_t2 = pd.merge(right=measures['hads_t2'].rename(columns={'anxiety': 'anxiety_t2', 'depression': 'depression_t2'}), 
                                    left=part_index, 
                                    left_on='t2', 
                                    right_on='B_Number', 
                                    how='left').drop(['index_x', 'B_Number',  'index_y', 'group', 't1'],  axis=1)
    replace_hads = hads_t2[~hads_t2['t2'].isin(measures['hads_post_break']['B_Number'])].reset_index(drop=True)
    replace_hads = replace_hads[replace_hads['t2'] != 'B2024b']
    repeated_measure = measures['hads_post_break'][['B_Number', 'anxiety', 'depression']].rename(columns={'B_Number': 't2', 
                                                                                        'anxiety': 'anxiety_t2', 'depression': 'depression_t2'})
    repeated_measure['t2'] = repeated_measure['t2'].str.rstrip()
    replace_hads['t2'] = replace_hads['t2'].str.rstrip()
    
    # Hads df could be concat rather than merged
    t2_hads = pd.concat((repeated_measure, replace_hads)).sort_values(by='t2').reset_index(drop=True)
    had = pd.concat((hads_df[['t1', 'anxiety_t1', 'depression_t1']], t2_hads), axis=1) 
    
    # Similar to hads/edeq however only one df this time. Filter participants who did not take part at t2
    time_df = pd.merge(right=measures['time_difference'], 
                                    left=part_index, 
                                    on='t1',  
                                    how='left').drop(['index_x','t2_y'], axis=1).rename(columns={'t2_x': 't2'})
    
    # Remove repeated measures and concat 
    replace_time = time_df[~time_df['t2'].isin(measures['time_post_break']['t2'])].reset_index(drop=True)
    replace_time = replace_time[replace_time['t2'] != 'B2024b']
    repeated_measure = measures['time_post_break']
    repeated_measure['t2'] = repeated_measure['t2'].str.rstrip()
    time = pd.concat((repeated_measure, replace_time)).sort_values(by='t2').reset_index(drop=True).drop(['index', 'index_y'], axis=1)
    
    # Manually set some missing values that where not on original dataframe
    time.loc[time['t2'].str.contains('B1009', na=False), 'years'] = 2.690411
    time.loc[time['t2'].str.contains('B2090', na=False), 'years'] = 2.284932
    time.loc[time['t2'].str.contains('B2091', na=False), 'years'] = 2.169863
    time.loc[time['t2'].str.contains('B2095', na=False), 'years'] = 3.6630137
    time = time.drop(['days', 'group'], axis=1)
    
    # Get age by adding years from time df to original age
    age = measures['age'][measures['age']['G_Number'].isin(time_df['t1'])][['G_Number', 'Age']].reset_index(drop=True)
    time_df_reordered = time.sort_values(by='t1')
    age['t2_age'] = age['Age'] + time_df_reordered['years']
    
    # BMI. Merge BMI onto the index to remove participants who didn't take part in neuroimaging
    
    bmi = pd.concat((pd.merge(ede_df[['t1']], measures['bmi_t1'][['BMI_baseline', 'G_Number']], 
                              left_on='t1', right_on='G_Number').drop('G_Number', axis=1),
                    measures['bmi_neuroimaging_t2']['bmi']), axis=1)
    
    # Get big df to be saved as sql 
    participant_data_for_neuroimaging = pd.concat((edeq[['t1', 't2', 'gs_t1', 'gs_t2']].rename(columns={'gs_t1': 'edeq_global_score_t1', 'gs_t2': 'edeq_global_score_t2'}), 
                                                   had[['anxiety_t1', 'anxiety_t2', 'depression_t1', 'depression_t2']], 
                                                   bmi[['BMI_baseline', 'bmi']].rename(columns={'BMI_baseline': 'bmi_t1', 'bmi': 'bmi_t2'}), 
                                                   time['years']), axis=1)
    # Merge in age
    participant_data_for_neuroimaging = pd.merge(participant_data_for_neuroimaging, age, 
                                                 right_on='G_Number', left_on='t1', 
                                                 how='left').drop('G_Number', axis=1).rename(columns={'Age': 'age_t1', 't2_age': 'age_t2'}).drop(index=93)
    
    # Get change scores
    participant_data_for_neuroimaging['edeq_change_score'] = participant_data_for_neuroimaging['edeq_global_score_t2'] - participant_data_for_neuroimaging['edeq_global_score_t1']
    participant_data_for_neuroimaging['anxiety_change_score'] = participant_data_for_neuroimaging['anxiety_t2'] - participant_data_for_neuroimaging['anxiety_t1']
    participant_data_for_neuroimaging['depression_change_score'] = participant_data_for_neuroimaging['depression_t2'] - participant_data_for_neuroimaging['depression_t1']
    participant_data_for_neuroimaging['bmi_change_score'] = participant_data_for_neuroimaging['bmi_t2'] - participant_data_for_neuroimaging['bmi_t1']
    
    # Save to database
    connector = connect_to_database('BEACON')
    participant_data_for_neuroimaging.to_sql('neuroimaging_behavioural_measures', connector)