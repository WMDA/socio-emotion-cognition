import argparse
import os
import json
import pandas as pd
import re

def options():
    flags = argparse.ArgumentParser()
    flags.add_argument('-d', '--dir', dest='dir', help='directory to search through')
    flags.add_argument('-s', '--save', dest='save', help='directory to save csvs to') 
    arg = vars(flags.parse_args())
    return arg

def root_file_names(file_dir):
    files = os.listdir(file_dir)
    return files

def get_filepaths(file_dir):
    
    dirs = root_file_names(file_dir)
    filter = [dir for dir in dirs if 'sub' in dir]
    anat_json = [file_dir + dir + '/' + dir + '/anat' for dir in filter]
    func_json = [file_dir + dir + '/' + dir + f'/func' for dir in filter]
    
    file_paths = {'anat':anat_json,
                  'func':func_json}
    
    return file_paths
    
def anat_json(file_paths):
    cnr = {'id':[],
           'cnr':[]}
    
    for sub in file_paths:
        file = os.listdir(sub)
        json_file = open(sub + '/' + file[0])
        data = json.load(json_file)
        cnr['id'].append(data['bids_meta']['subject_id'])
        cnr['cnr'].append(data['cnr'])
        
    return cnr


def func_json(file_path):
    jsons = ['_task-eft_bold.json', '_task-fear_bold.json', '_task-happy_bold.json', '_task-rest_run-01_bold.json', '_task-rest_run-02_bold.json',  '_task-rest_run-03_bold.json']
    
    info = {
            'id': [], 
            'task': [],
            'snr': [],
            'fd_mean': [],
            'fd_num': [],
            'fd_perc': []
            }

    for sub in file_path:
        part = re.findall(r'sub-B.....', sub)
        part = part[0].rstrip('/')

        for dat_file in jsons:
            json_file = open(sub + '/' + part + dat_file)
            data = json.load(json_file)
            info['id'].append(data['bids_meta']['subject_id'])
            info['task'].append(data['bids_meta']['SeriesDescription'])
            info['snr'].append(data['snr'])
            info['fd_mean'].append(data['fd_mean'])
            info['fd_num'].append(data['fd_num'])
            info['fd_perc'].append(data['fd_perc'])
                    
            
    return info
  
if __name__ == '__main__':
    
    flags = options()
    paths = get_filepaths(flags['dir'])
    cnr = anat_json(paths['anat'])
    ant_df = pd.DataFrame(cnr)
    func_var = func_json(paths['func'])
    func_df = pd.DataFrame(func_var)
    
    tasks = func_df.groupby('task')
    happy_df = tasks.get_group('happy v neutral').sort_values(by=['id']).reset_index(drop=True)
    fear_df = tasks.get_group('fear v neutral').sort_values(by=['id']).reset_index(drop=True)
    eft_df = tasks.get_group('emdedded figs').sort_values(by=['id']).reset_index(drop=True)
    rest_df = tasks.get_group('Multi-Echo rsFMRI').sort_values(by=['id']).reset_index(drop=True)
    
    happy_df.to_csv(flags['save'] + 'task_happy.csv')
    fear_df.to_csv(flags['save'] + 'task_fear.csv')
    eft_df.to_csv(flags['save'] + 'task_eft.csv')
    rest_df.to_csv(flags['save'] + 'task_rest.csv')
    ant_df.to_csv(flags['save'] + 'anat.csv')
