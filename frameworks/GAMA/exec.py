import logging
import os
import sys
import tempfile as tmp

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import category_encoders
from packaging import version
import sklearn

from gama.data_loading import file_to_pandas
from gama import GamaClassifier, GamaRegressor, __version__

from frameworks.shared.callee import call_run, result, save_metadata, utils


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** GAMA [v%s] ****", __version__)
    log.info("sklearn == %s", sklearn.__version__)
    log.info("category_encoders == %s", category_encoders.__version__)

    is_classification = (config.type == 'classification')
    # Mapping of benchmark metrics to GAMA metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        mse='neg_mean_squared_error',
        msle='neg_mean_squared_log_error',
        r2='r2',
        rmse='neg_mean_squared_error',
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    *_, did, fold = dataset.train_path.split('/')
    fold = fold.split('.')[0].split('_')[-1]

    log.info('Running GAMA with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)

    estimator = GamaClassifier if is_classification else GamaRegressor
    kwargs = dict(
        n_jobs=n_jobs,
        max_total_time=config.max_runtime_seconds,
        scoring=scoring_metric,
        random_state=config.seed,
        **training_params
    )
    version_leq_20_2_0 = version.parse(__version__) <= version.parse('20.2.0')
    if version_leq_20_2_0:
        log_file = os.path.join(config.output_dir, "logs", '{}_{}.log'.format(did, fold))
        utils.touch(log_file)
        kwargs['keep_analysis_log'] = log_file
    else:
        kwargs['max_memory_mb'] = config.max_mem_size_mb
        kwargs['output_directory'] = os.path.join(config.output_dir, "gama")
    
    gama_automl = estimator(**kwargs)

    data = file_to_pandas(dataset.train_path, encoding='utf-8')
    x, y = data.loc[:, data.columns != dataset.target], data.loc[:, dataset.target]

    with utils.Timer() as training_timer:
        gama_automl.fit(x, y)

    data = file_to_pandas(dataset.test_path, encoding='utf-8')
    x, _ = data.loc[:, data.columns != dataset.target], data.loc[:, dataset.target]
    log.info('Predicting on the test set.')
    with utils.Timer() as predict_timer:
        predictions = gama_automl.predict(x)
    if is_classification:
        probabilities = gama_automl.predict_proba(x)
    else:
        probabilities = None

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        probabilities=probabilities,
        target_is_encoded=False,
        models_count=len(gama_automl._final_pop),
        training_duration=training_timer.duration,
        predict_duration=predict_timer.duration
    )


if __name__ == '__main__':
    call_run(run)
