import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pipeline engine
import nipype.algorithms.modelgen as model
import nipype.interfaces.spm as spm  # spm interface
from nipype.interfaces.io import DataSink
from nipype.algorithms.misc import Gunzip

# Non nipye libaries
import os
from bids.layout import BIDSLayout
import sys
from decouple import config

def subjectinfo(subject_id: str) -> list:

    '''
    Function to define subjects. Gets regressors and events and returns nipype 
    Bunch object. Nipype needs libaries to be declared in the function to be 
    able to run. 

    Parameters
    ----------
    id: str of subject id
    events_df: DataFrame of events.
    confounds_df: DataFrame of counfounds from fMRIPrep output.
    
    Returns
    -------
    list: nipype Bunch object
    '''

    from decouple import config
    import os
    import pandas as pd
    from nipype.interfaces.base import Bunch
    
    experiment_dir = config('happy')
    dfs_events = os.path.join(config('raw_data'), 'bids_t1')
    events_df = pd.read_csv(dfs_events + f'/{subject_id}/func/{subject_id}_task-happy_events.tsv')
    confounds_df = pd.read_csv(os.path.join(experiment_dir,'preprocessed_t1', subject_id) +
                        f'/func/{subject_id}_task-happy_desc-confounds_timeseries.tsv', sep='\t')

    regres = {
        'trans_x': confounds_df['trans_x'],
        'trans_y': confounds_df['trans_y'],
        'trans_z': confounds_df['trans_z'],
        'rot_x': confounds_df['rot_x'],
        'rot_y': confounds_df['rot_y'],
        'rot_z': confounds_df['rot_z']
    }

    events = {
        'blank': events_df[events_df['trial_type'].str.contains('Blank')],
        'neutral': events_df[events_df['trial_type'].str.contains('Neutral')],
        'partially_happy': events_df[events_df['trial_type'].str.contains('Partially_happy')],
        'happy': events_df[events_df['trial_type'] == 'happy']
    }
        
    subject_info =  [
        Bunch(conditions=[condition for condition in events.keys()],
            onsets=[events[key]['onset'].to_list() for key in events.keys()],
            durations=[events[key]['duration'].to_list() for key in events.keys()],
            regressor_names=[regess_name for regess_name in regres.keys()],
            regressors=[regres[key].to_list() for key in regres.keys()],
            )]

    return subject_info

#  Global parameters lots of these are repeated due to nipype quirks
experiment_dir = config('happy')
subject_to_analyse = [sys.argv[1]] 
layout = BIDSLayout(os.path.join(experiment_dir,'preprocessed_t1', subject_to_analyse[0]), validate=False)
img_file = layout.get(subject=subject_to_analyse[0].lstrip('sub-'), datatype='func', 
                      space='MNI152NLin2009cAsym', suffix='bold', extension='nii.gz')[0]
img_file_path = os.path.join(layout.root,'func', img_file.filename)
base_dir = os.path.join(experiment_dir, '1stlevel')

#  Building the contrasts
contrasts = [['linear_contrast', 'T', ['blank', 'neutral', 'partially_happy', 'happy'], 
             [0, -1, 0, 1, 0, 0, 0, 0, 0, 0]]]

#  Building the SPM workflow
level_1_analysis = pe.Workflow(name='analysis')  
level_1_analysis.base_dir = os.path.join(base_dir, 'workingdir')

#  Build the generic model node
model_spec = pe.Node(model.SpecifySPMModel(), name='modelspec') 
model_spec.inputs.input_units = 'scans'
model_spec.inputs.output_units = 'scans'
model_spec.inputs.time_repetition = 2.0
model_spec.inputs.high_pass_filter_cutoff = 128
model_spec.inputs.concatenate_runs = False

#  Node to define the SPM first level model
level_1_design = pe.Node(interface=spm.Level1Design(), name='fMRIModelspecs') 
level_1_design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level_1_design.inputs.model_serial_correlations = 'FAST'
level_1_design.inputs.interscan_interval = 2.0
level_1_design.inputs.timing_units = 'scans'
level_1_design.inputs.global_intensity_normalization = 'none'
level_1_design.inputs.use_mcr  = False
level_1_design.inputs.volterra_expansion_order = 1
level_1_design.inputs.microtime_onset = 20
level_1_design.inputs.microtime_resolution = 41
level_1_design.inputs.flags = {'mthresh': 0.8}

#  Node to estimate the model
level_1_estimate = pe.Node(interface=spm.EstimateModel(), name='level1Estimate')  
level_1_estimate.inputs.estimation_method = {'Classical': 1}  
level_1_estimate.inputs.write_residuals = False
level_1_estimate.inputs.use_mcr = False

#  Node to estimate the contrasts
contrast_estimates = pe.Node(spm.EstimateContrast(), name='contrastestimates')  

#  Nodes to get subjects data (events and confounds)
subject = pe.Node(util.Function(input_names=['subject_id'],
                                output_names=['subject_info'],
                                function=subjectinfo,
                               ),
                  name='getsubjectinfo')  

#  Node to iterate over subject names
info_source = pe.Node(util.IdentityInterface(fields=['subject_id',
                                                     'contrasts'],
                                             contrasts=contrasts),
                      name='info_source')
info_source.inputs.gunzip = 'decompress'
info_source.iterables = [('subject_id', subject_to_analyse)]

#  Creates the output folders
data_sink = pe.Node(DataSink(base_directory=base_dir,
                             container=base_dir),
                    name='datasink')
substitutions = [('_subject_id_', '')]
data_sink.inputs.substitutions = substitutions

#  Create gunzip node to gunzip nii.gz scans
gunzip = pe.Node(Gunzip(in_file=img_file_path), name="gunzip")

#  Define the workflow and nodes
level_1_analysis.connect([
(info_source, gunzip, [('gunzip', 'mode')]),
(info_source, subject, [('subject_id', 'subject_id')]),
(subject, model_spec, [('subject_info', 'subject_info')]),
(info_source, contrast_estimates, [('contrasts', 'contrasts')]),
(gunzip, model_spec, [('out_file', 'functional_runs')]),
(model_spec, level_1_design, [('session_info', 'session_info')]),
(level_1_design, level_1_estimate, [('spm_mat_file', 'spm_mat_file')]),
(level_1_estimate, contrast_estimates, [('spm_mat_file', 'spm_mat_file'), 
                                        ('beta_images', 'beta_images'),
                                        ('residual_image', 'residual_image')]),
(contrast_estimates, data_sink, [('spm_mat_file', 'T1.@spm_mat'),
                                     ('spmT_images', 'T1.@T'),
                                     ('con_images', 'T1.@con')]),
(level_1_estimate, data_sink, [('beta_images', 'T1.@beta')])
])

if __name__ == "__main__":
    level_1_analysis.write_graph(graph2use='colored', format='png', simple_form=True)
    level_1_analysis.run()
    print('\n\n','-'*100)
    print(f'\nCompleted 1st Level modelling for {subject_to_analyse[0]}\n')
    print('Cleaning up workingdir')
    working_path = os.path.join(base_dir, 'workingdir', 'analysis', f'_subject_id_{subject_to_analyse[0]}')
    print(f'Removing {working_path}')
    os.system(f'rm -rf {working_path}')