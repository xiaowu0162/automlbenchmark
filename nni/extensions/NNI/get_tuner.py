from amlb.benchmark import TaskConfig

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner



def get_tuner(config: TaskConfig):
    if 'tuner_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "tuner_type" field.')

    # add new Tuners here 
    if config.framework_params['tuner_type'] == 'tpe':
        return HyperoptTuner('tpe'), 'TPE Tuner'

    elif config.framework_params['tuner_type'] == 'random_search':
        return HyperoptTuner('random_search'), 'Random Search Tuner'

    elif config.framework_params['tuner_type'] == 'anneal':
        return HyperoptTuner('anneal'), 'Annealing Tuner'
    
    elif config.framework_params['tuner_type'] == 'evolution':
        return EvolutionTuner(), 'Evolution Tuner'
    
    else:
        raise RuntimeError('The requested tuner type in framework.yaml is unavailable.')

