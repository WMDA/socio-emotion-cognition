import os 
import re
from decouple import config
import glob

def rename_fmri(path):
    files = glob.glob(f'{path}/*.nii*')

    for file in files:
        remove_prefix_name = re.sub(r'mixed_model_', '' , file)
        fwep = re.sub(r'tstat_fwep', 'fwep' , remove_prefix_name)
        time = re.sub(r'c1', 'time' , fwep)
        interaction = re.sub(r'c2', 'interaction', time)
        print(interaction)
        os.rename(file, interaction)

def rename_eft_files(path):
    files - glob.glob(f'{path}/*._task-eft_desc-confounds_regressors.tsv')
    for file in files:
       name = re.sub('_regressors', '_timeseries', file)
       print(name)
       #os.rename(file, name)




path = os.path.join(config('eft'), '2ndlevel', 'mixed_model')
rename_fmri(path)

