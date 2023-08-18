from fNeuro.behavioural.data_functions import load_data
import pandas as pd
from decouple import config
from sqlalchemy import create_engine

tables =["aq10_t2",
"bmi_neuroimaging",
"bmi_t1",
"bmi_t2",
"edeq_post_break",
"edeq_t1",
"edeq_t2",
"hads_post_break",
"hads_t1",
"hads_t2",
"neuroimaging_behavioural_measures",
"neuroimaging_index",
"oci_t1",
"oci_t2",
"participant_index",
"pca_df",
"pca_t1",
"pca_t2",
"raw_edeq_t1",
"raw_hads_t1",
"raw_t1",
"raw_t2",
"raw_t2_all_values",
"raw_t2_question_index",
"t1_measures",
"t2_measures",
"time_difference",
"time_post_break",
"wsas_t1",
"wsas_t2"]

username = config('cloud_username').rstrip()
password = config('cloud_password').rstrip()
cloud=config('cloud')
connector = create_engine(
        f'mysql+mysqlconnector://{username}:{password}@{cloud}/BEACON')

for table in tables:
    table_data = load_data('BEACON', table)
    table_data.to_sql(table, connector)