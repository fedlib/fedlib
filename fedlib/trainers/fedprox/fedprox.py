from functools import partial

import torch
from ray.rllib.utils.annotations import override
from fedlib.clients.callbacks import ClientCallback
from fedlib.trainers.fedavg import (
    FedavgTrainer,
    FedavgTrainerConfig,
)


class FedProxClientCallback(ClientCallback):
    def __init__(self, prox_mu):
        """Initializes a FedProxClientCallback instance."""
        super().__init__()
        self.prox_mu = prox_mu

    @override(ClientCallback)
    def on_backward_begin(self, loss, task):
        """Called when the backward pass begins."""
        global_model_state = task.global_model_state_dict
        model_states = task.model.state_dict()
        if self.prox_mu > 0:
            for local_param, global_param in zip(
                global_model_state.values(), model_states.values()
            ):
                if local_param.requires_grad:
                    loss += (
                        0.5 * self.prox_mu * torch.norm(local_param - global_param) ** 2
                    )
        return loss


class FedProxTrainerConfig(FedavgTrainerConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedProxTrainerConfig instance."""
        super().__init__(algo_class=algo_class or FedProxTrainer)

        self.prox_mu = 0.1

    @override(FedavgTrainerConfig)
    def get_client_config(self):
        partial_callback = partial(FedProxClientCallback, prox_mu=self.prox_mu)
        config = super().get_client_config().callbacks(partial_callback)
        return config


class FedProxTrainer(FedavgTrainer):
    pass
