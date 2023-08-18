from decouple import config
import os
import sys
from nilearn.decoding import SpaceNetRegressor
from sklearn.model_selection import GridSearchCV
from fNeuro.MVPA.mvpa_functions import save_pickle, ados
import numpy as np

if __name__ == "__main__":
    
    tv_l1_path = config('ml')
    print('\nBuilding TV-l1 models')
    print('\nReading ADOS results')
    
    ados_df = ados('G2', test_train='train', directory='combined')
    ados_df = ados_df.drop([20]) # Remove the one outlier
    models = {}
    for domain in ados_df.columns[1:6]:
        print(f'\nWorking on {domain}')
        tv_l1 = SpaceNetRegressor(penalty="tv-l1", 
                                  l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                  n_jobs=8,
                                  cv=10)
        tv_l1.fit(ados_df['paths'], ados_df[domain])
        models[domain] = tv_l1
    try:
        print('\nSaving output')
        save_pickle(os.path.join(tv_l1_path, 'spacenet_best_estimator', 'spacenet_models'), models)
    except Exception as e:
        print(e)
        sys.exit(1)
    print('\nFinished calculating spacenet models')