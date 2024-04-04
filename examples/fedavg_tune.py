import ray
from fedlib.trainers.fedavg import FedavgTrainerConfig, FedavgTrainer
from ray import tune
from ray.tune.stopper import MaximumIterationStopper

from fedlib.trainers import TrainerConfig


class ExampleFedavgConfig(FedavgTrainerConfig):
    def __init__(self, algo_class=None):
        """Initialize a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or ExampleFedavg)

        self.dataset_config = {
            "type": "FashionMNIST",
            "num_clients": 10,
            "train_batch_size": 32,
        }
        self.global_model = "cnn"


class ExampleFedavg(FedavgTrainer):
    def __init__(self, config=None, logger_creator=None, **kwargs):
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> TrainerConfig:
        return ExampleFedavgConfig()


if __name__ == "__main__":
    ray.init()

    config_dict = (
        ExampleFedavgConfig()
        .resources(
            num_gpus_for_driver=0.0,
            num_cpus_for_driver=1,
            num_remote_workers=0,
            num_gpus_per_worker=0.0,
        )
        .to_dict()
    )
    print(config_dict)
    tune.run(
        ExampleFedavg,
        config=config_dict,
        stop=MaximumIterationStopper(100),
    )
