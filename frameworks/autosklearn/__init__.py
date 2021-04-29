from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X=dataset.train.X_enc,
            y=dataset.train.y_enc
        ),
        test=dict(
            X=dataset.test.X_enc,
            y=dataset.test.y_enc
        ),
        predictors_type=['Numerical' if p.is_numerical() else 'Categorical' for p in dataset.predictors]
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

