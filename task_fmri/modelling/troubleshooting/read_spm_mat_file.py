from scipy.io import loadmat

mat_file = loadmat('/data/project/BEACONB/task_fmri/eft/1stlevel/T2/sub-B2999/SPM.mat')
print(mat_file['SPM']['xX'][0][0])#['X'])
import sys
sys.exit(0)
import pandas as pd
df = pd.DataFrame(data=mat_file['SPM']['xX'][0][0]['X'][0][0])
print(df)

