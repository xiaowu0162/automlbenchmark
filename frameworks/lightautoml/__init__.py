from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            data=dataset.train.data
        ),
        test=dict(
            X=dataset.test.X,
            y=dataset.test.y
        ),
        target=dict(
            name=dataset.target.name,
        ),
        problem_type=dataset.type.name
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)
