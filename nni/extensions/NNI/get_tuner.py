from amlb.benchmark import TaskConfig

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.smac_tuner.smac_tuner import SMACTuner
from nni.algorithms.hpo.gp_tuner.gp_tuner import GPTuner
from nni.algorithms.hpo.metis_tuner.metis_tuner import MetisTuner


def get_tuner(config: TaskConfig):
    if 'tuner_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "tuner_type" field.')

    # Users may add their customized Tuners here 
    if config.framework_params['tuner_type'] == 'tpe':
        return HyperoptTuner('tpe'), 'TPE Tuner'

    elif config.framework_params['tuner_type'] == 'random_search':
        return HyperoptTuner('random_search'), 'Random Search Tuner'

    elif config.framework_params['tuner_type'] == 'anneal':
        return HyperoptTuner('anneal'), 'Annealing Tuner'
    
    elif config.framework_params['tuner_type'] == 'evolution':
        return EvolutionTuner(), 'Evolution Tuner'

    elif config.framework_params['tuner_type'] == 'smac':
        return SMACTuner(), 'SMAC Tuner'

    elif config.framework_params['tuner_type'] == 'gp':
        return GPTuner(), 'GP Tuner'

    elif config.framework_params['tuner_type'] == 'metis':
        return MetisTuner(), 'Metis Tuner'
    
    # TO-DO: Hyperband, NetworkMorphism, BOHB, PPO, PBT
    
    # Note: BatchTuner and GridSearchTuner are not included
    else:
        raise RuntimeError('The requested tuner type in framework.yaml is unavailable.')

