import os 
import re
from decouple import config
import glob

path = os.path.join(config('eft'), '2ndlevel', 'mixed_model')

files = glob.glob(f'{path}/*.nii*')

for file in files:
    remove_prefix_name = re.sub(r'mixed_model_', '' , file)
    fwep = re.sub(r'tstat_fwep', 'fwep' , remove_prefix_name)
    time = re.sub(r'c1', 'time' , fwep)
    interaction = re.sub(r'c2', 'interaction', time)
    print(interaction)
    os.rename(file, interaction)

