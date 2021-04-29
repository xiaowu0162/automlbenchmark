import logging
import sklearn
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from nni.algorithms.hpo.evolution_tuner import EvolutionTuner

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.results import save_predictions_to_file
from amlb.utils import Timer

log = logging.getLogger(__name__)


SEARCH_SPACE = {
    "n_estimators": {"_type":"choice", "_value": [128, 256, 512, 1024, 2048, 4096]},
    "max_depth": {"_type":"choice", "_value": [5, 10, 25, 50, 100]},
}

    
def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Random Forest (sklearn) Tuned with NNI EvolutionTuner ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute(dataset.train.X, dataset.test.X)
    y_train, y_test = dataset.train.y, dataset.test.y

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor
    # model = estimator(random_state=config.seed, **config.framework_params)
    best_score, best_params, best_model = None, None, None
    score_higher_better = True

    log.info("Tuning hyperparameters with NNI EvolutionTuner with a maximum time of {}s\n"
             .format(config.max_runtime_seconds))
    tuner = EvolutionTuner()
    tuner.update_search_space(SEARCH_SPACE)
    start_time = time.time()
    param_idx = 0
    while True:
        try:
            cur_params = tuner.generate_parameters(param_idx)
            cur_model = estimator(random_state=config.seed, **cur_params, **config.framework_params)
            # Here score is the output of score() from the estimator
            cur_score = cross_val_score(cur_model, X_train, y_train)
            cur_score = sum(cur_score) / float(len(cur_score))
            if best_score is None
            or (score_higher_better and cur_score > best_score)
            or (not score_higher_better and cur_score < best_score):
                best_score, best_params, best_model = cur_score, cur_params, cur_model
                
            log.info("Trial {}: \n{}\nScore: {}\n".format(param_idx, cur_params, cur_score))
            tuner.receive_trial_result(param_idx, cur_params, cur_score)
            param_idx += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > config.max_runtime_seconds:
                break
        except:
            break
        
    log.info("Tuning done, the best parameters are:\n{}\n".format(best_params))

    # retrain on the whole dataset 
    with Timer() as training:
        best_model.fit(X_train, y_train)     
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test) if is_classification else None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test)

    return dict(
        models_count=1,
        training_duration=training.duration
    )
