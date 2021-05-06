import nni

from nni.utils import MetricType 

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.smac_tuner.smac_tuner import SMACTuner
from nni.algorithms.hpo.gp_tuner.gp_tuner import GPTuner
from nni.algorithms.hpo.metis_tuner.metis_tuner import MetisTuner
from nni.algorithms.hpo.hyperband_advisor import Hyperband


def get_tuner(tuner_alias):
    # Users may add their customized Tuners here 
    if tuner_alias == 'tpe':
        return HyperoptTuner('tpe'), 'TPE Tuner'

    elif tuner_alias == 'random_search':
        return HyperoptTuner('random_search'), 'Random Search Tuner'

    elif tuner_alias == 'anneal':
        return HyperoptTuner('anneal'), 'Annealing Tuner'
    
    elif tuner_alias == 'evolution':
        return EvolutionTuner(), 'Evolution Tuner'

    elif tuner_alias == 'smac':
        return SMACTuner(), 'SMAC Tuner'

    elif tuner_alias == 'gp':
        return GPTuner(), 'GP Tuner'

    elif tuner_alias == 'metis':
        return MetisTuner(), 'Metis Tuner'

    elif tuner_alias == 'hyperband':
        return Hyperband(), 'Hyperband Advisor'
    
    # TO-DO: BOHB
    
    # Note: BatchTuner and GridSearchTuner are not included
    else:
        raise RuntimeError('The requested tuner type in framework.yaml is unavailable.')

    
class NNITuner:
    '''
    A specialized wrapper for the automlbenchmark framework.
    Abstracts the different behaviors of tuners and advisors into a tuner API. 
    '''
    def __init__(self, tuner_alias):
        self.core, self.description = get_tuner(tuner_alias)

        self.core_type = None      # 'tuner' or 'advisor'   
        if isinstance(self.core, nni.tuner.Tuner):
            self.core_type = 'tuner'
        elif isinstance(self.core, nni.runtime.msg_dispatcher_base.MsgDispatcherBase):
            self.core_type = 'advisor'
        else:
            raise RuntimeError('Unsupported tuner or advisor type') 

        # note: tuners and advisors use this variable differently
        self.cur_param_id = 0

        
    def __del__(self):
        self.handle_terminate()

        
    def update_search_space(self, search_space):
        if self.core_type == 'tuner':
            return self.core.update_search_space(search_space)
            
        elif self.core_type == 'advisor':
            #return self.core.handle_update_search_space(search_space)
            return self.core.handle_initialize(search_space)

        
    def generate_parameters(self):
        self.cur_param_id += 1
        if self.core_type == 'tuner':
            self.cur_param = self.core.generate_parameters(self.cur_param_id-1)
            return self.cur_param_id-1, self.cur_param
            
        elif self.core_type == 'advisor':
            self.cur_param = self.core._get_one_trial_job()
            hyperparams = self.cur_param['parameters'].copy()
            hyperparams.pop('TRIAL_BUDGET')
            return self.cur_param['parameter_id'], hyperparams

        
    def receive_trial_result(self, parameter_id, parameters, value):
        if self.core_type == 'tuner':
            return self.core.receive_trial_result(parameter_id, parameters, value)

        elif self.core_type == 'advisor':
            metric_report = {}
            metric_report['parameter_id'] = parameter_id
            metric_report['trial_job_id'] = self.cur_param_id
            metric_report['type'] = MetricType.FINAL
            metric_report['value'] = str(value)
            metric_report['sequence'] = self.cur_param_id
            return self.core.handle_report_metric_data(metric_report)   

        
    def handle_terminate(self):
        if self.core_type == 'tuner':
            pass
        
        elif self.core_type == 'advisor':   
            self.core.stopping = True 

    
