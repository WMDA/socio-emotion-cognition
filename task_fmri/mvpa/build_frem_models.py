from decouple import config
import os
import sys
from nilearn.decoding import FREMRegressor
from mvpa_functions import save_pickle, ados
from sklearn.svm import SVR 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    frem_path = config('ml')
    print('\nBuilding FREM models')
    print('\nReading ADOS results')
    
    ados_df = ados('G2', test_train='train', directory='combined')
    ados_df = ados_df.drop([20]) # Remove the one outlier
    for domain in ados_df.columns[1:6]:
        print(f'\nWorking on {domain}')
        frem = FREMRegressor(estimator=SVR(kernel='linear'), n_jobs=8)
        frem.fit(ados_df['paths'], ados_df[domain])
        try:
            print('\nSaving output')
            save_pickle(os.path.join(frem_path, 'frem', domain), frem)
        except Exception as e:
            print(e)
            sys.exit(1)
    print('\nFinished calculating FREM models')