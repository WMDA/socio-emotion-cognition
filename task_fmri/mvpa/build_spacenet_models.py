from decouple import config
import os
import sys
from nilearn.decoding import SpaceNetRegressor
from mvpa_functions import save_pickle, ados

if __name__ == "__main__":
    
    tv_l1_path = config('ml')
    print('\nBuilding TV-l1 models')
    print('\nReading ADOS results')
    
    ados_df = ados('G2', mean_images=True)
    for domain in ados_df.columns[1:6]:
        print(f'\nWorking on {domain}')
        tv_l1 = SpaceNetRegressor(penalty="tv-l1", 
                                  eps=1e-1, 
                                  n_jobs=8)
        tv_l1.fit(ados_df['paths'], ados_df[domain])
        try:
            print('\nSaving output')
            save_pickle(os.path.join(tv_l1_path, 'pickle', 'combined', domain), tv_l1)
        except Exception as e:
            print(e)
            sys.exit(1)
        
    print('\nFinished calculating FREM models')