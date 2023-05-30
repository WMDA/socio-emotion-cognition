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
    able to run. Large Function but quirk of nipype makes it difficult to 
    reduce in size.

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
    
    experiment_dir = config('eft')
    dfs_events = os.path.join(config('raw_data'), 'bids_t2')
    events_df = pd.read_csv(dfs_events + f'/{subject_id}/func/{subject_id}_task-eft_events.tsv', sep=',')
    confounds_df = pd.read_csv(os.path.join(experiment_dir,'preprocessed_t2', subject_id) +
                        f'/func/{subject_id}_task-eft_desc-confounds_timeseries.tsv', sep='\t').fillna(0)
    
    # Data driven PCA approach. Selects the components to exaplain the minimum cumulative fraction of variance
    regres = confounds_df.filter(regex=("a_comp_cor.*")).to_dict(orient='series')
    
    # Then add in the movement regressors
    regres.update([            
        ('trans_x_derivative1', confounds_df['trans_x_derivative1']),
        ('trans_y_derivative1', confounds_df['trans_y_derivative1']),
        ('trans_z_derivative1', confounds_df['trans_z_derivative1']),
        ('rot_x_derivative1', confounds_df['rot_x_derivative1']),
        ('rot_y_derivative1', confounds_df['rot_y_derivative1']),
        ('rot_z_derivative1', confounds_df['rot_z_derivative1']), 
        ('trans_x_power2', confounds_df['trans_x_power2']),
        ('trans_y_power2', confounds_df['trans_y_power2']),
        ('trans_z_power2', confounds_df['trans_z_power2']),
        ('rot_x_power2', confounds_df['rot_x_power2']),
        ('rot_y_power2', confounds_df['rot_y_power2']),
        ('rot_z_power2', confounds_df['rot_z_power2']), 
        ('trans_x_derivative1_power2', confounds_df['trans_x_derivative1_power2']),
        ('trans_y_derivative1_power2', confounds_df['trans_y_derivative1_power2']),
        ('trans_z_derivative1_power2', confounds_df['trans_z_derivative1_power2']),
        ('rot_x_derivative1_power2', confounds_df['rot_x_derivative1_power2']),
        ('rot_y_derivative1_power2', confounds_df['rot_y_derivative1_power2']),
        ('rot_z_derivative1_power2', confounds_df['rot_z_derivative1_power2']),
        ])

    events = {
        'Baseline': events_df[events_df['Condition'].str.contains('Baseline')],
        'ComplexFigures': events_df[events_df['Condition'].str.contains('ComplexFigures')],
        'SimpleFigures': events_df[events_df['Condition'].str.contains('SimpleFigures')],
    }
    
    
    events_df['RT'] = events_df['RT'].apply(lambda rt: int(0) if rt =='.' else int(rt))
    events_df['demean_rt'] = events_df['RT'] -  events_df['RT'].mean()  
    events_df['correct'] = events_df['IsCorrect'].apply(lambda correct: 1 if correct =='Yes' else 0)

    subject_info =  [
        Bunch(conditions=[condition for condition in events.keys()],
            onsets=[events[key]['onset'].to_list() for key in events.keys()],
            durations=[events[key]['duration'].to_list() for key in events.keys()],
            regressor_names=[regess_name for regess_name in regres.keys()],
            regressors=[regres[key].to_list() for key in regres.keys()],
            pmod=[
                Bunch(name=['RT'], param=[events_df['demean_rt'][events_df['Condition'].str.contains(condition)].tolist()], poly=[1]) 
                for condition in events.keys()    
            ]
            )]

    return subject_info

#  Global parameters lots of these are repeated due to nipype quirks
experiment_dir = config('eft')
subject_to_analyse = [sys.argv[1]] 
layout = BIDSLayout(os.path.join(experiment_dir,'preprocessed_t2', subject_to_analyse[0]), validate=False)
img_file = layout.get(subject=subject_to_analyse[0].lstrip('sub-'), datatype='func', 
                      space='MNI152NLin2009cAsym', suffix='bold', extension='nii.gz')[0]
img_file_path = os.path.join(layout.root,'func', img_file.filename)
base_dir = os.path.join(experiment_dir, '1stlevel')

#  Building the contrasts
t_contrast_canonical_mod = ['complex-simple_canonical_mod', 'T',  
                        ['ComplexFiguresxRT^1*bf(1)', 'SimpleFiguresxRT^1*bf(1)'], 
                        [1, -1]]

t_contrast_time_mod = ['complex-simple_time_mod', 'T',
                         ['ComplexFiguresxRT^1*bf(2)', 'SimpleFiguresxRT^1*bf(2)'], 
                         [1, -1]]

t_contrast_dispers_mod = ['complex-simple_dispers_mod', 'T',
                         ['ComplexFiguresxRT^1*bf(3)', 'SimpleFiguresxRT^1*bf(3)'], 
                         [1, -1]]

t_contrast_canonical =  ['complex-simple_canonical', 'T',  
                        ['ComplexFigures*bf(1)', 'SimpleFigures*bf(1)'], 
                        [1, -1]]

t_contrast_time = ['complex-simple_time', 'T',
                         ['ComplexFigures*bf(2)', 'SimpleFigures*bf(2)'], 
                         [1, -1]]

t_contrast_dispers = ['complex-simple_dispers', 'T',
                         ['ComplexFigures*bf(3)', 'SimpleFigures*bf(3)'], 
                         [1, -1]]

f_effect_of_interest_mod = ['effect_of_interest', 'F', [t_contrast_canonical_mod, t_contrast_time_mod, t_contrast_dispers_mod]]

contrasts = [t_contrast_canonical_mod, t_contrast_time_mod, t_contrast_dispers_mod, f_effect_of_interest_mod, 
             t_contrast_canonical, t_contrast_time, t_contrast_dispers]

#  Building the SPM workflow
level_1_analysis = pe.Workflow(name='analysis')  
level_1_analysis.base_dir = os.path.join(base_dir, 'workingdir')

#  Build the generic model node
model_spec = pe.Node(model.SpecifySPMModel(), name='modelspec') 
model_spec.inputs.input_units = 'secs'
model_spec.inputs.time_repetition = 2.0
model_spec.inputs.high_pass_filter_cutoff = 128
model_spec.inputs.concatenate_runs = False

#  Node to define the SPM first level model
level_1_design = pe.Node(interface=spm.Level1Design(), name='fMRIModelspecs') 
level_1_design.inputs.bases = {'hrf': {'derivs': [1, 1]}}
level_1_design.inputs.model_serial_correlations = 'FAST'
level_1_design.inputs.interscan_interval = 2.0
level_1_design.inputs.timing_units = 'secs'
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
contrast_estimates.inputs.use_derivs = True

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
(contrast_estimates, data_sink, [('spm_mat_file', 'T2.@spm_mat'),
                                     ('spmT_images', 'T2.@T'),
                                     ('con_images', 'T2.@con')]),
(level_1_estimate, data_sink, [('beta_images', 'T2.@beta')])
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