import warnings
from typing import List, TYPE_CHECKING

from ray.tune.callback import _CallbackMeta

if TYPE_CHECKING:
    from fedlib.trainers import Trainer


class TrainerCallback(metaclass=_CallbackMeta):
    """Abstract base class for Fedlib callbacks (similar to Keras callbacks).

    These callbacks can be used for custom metrics and custom postprocessing.

    By default, all of these callbacks are no-ops. To configure custom training
    callbacks, subclass DefaultCallbacks and then set
    {"callbacks": YourCallbacksClass} in the algo config.
    """

    def __init__(self) -> None:
        self._trainer = None

    def setup(
        self,
        trainer: "Trainer",
        **info,
    ):
        self._trainer = trainer

    def on_trainer_init(self, *, trainer: "Trainer", **kwargs):
        """Called at the end of Trainer.__init__.

        Subclasses should override for any actions to run.
        Args:
            trainer (Trainer): The Trainer object.
        """

    def on_train_round_begin(self) -> None:
        """Called at the beginning of each local training round in
        `train_global_model` methods.

        Subclasses should override for any actions to run.
        Returns:
        """

    def on_train_round_end(self):
        """A callback method called after local training.

        It is typically used to modify updates (i.e,. pseudo-gradient).
        """


class TrainerCallbackList:
    def __init__(self, callbacks: List[TrainerCallback]):
        self._callbacks = callbacks
        self._trainer = None

    def __getitem__(self, item):
        return self._callbacks[item]

    def setup(self, trainer, **info):
        self._trainer = trainer
        for callback in self._callbacks:
            try:
                callback.setup(trainer, **info)
            except TypeError as e:
                if "argument" in str(e):
                    warnings.warn(
                        "Please update `setup` method in callback "
                        f"`{callback.__class__}` to match the method signature"
                        " in `ray.tune.callback.Callback`.",
                        FutureWarning,
                    )
                    callback.setup(trainer)
                else:
                    raise e

    def append(self, callback: TrainerCallback):
        callback.setup(self._trainer)
        self._callbacks.append(callback)

    def on_trainer_init(self, *, trainer: "Trainer", **kwargs):
        for callback in self._callbacks:
            callback.on_trainer_init(trainer=trainer, **kwargs)

    def on_train_round_begin(self) -> None:
        for callback in self._callbacks:
            callback.on_train_round_begin()

    def on_train_round_end(self):
        for callback in self._callbacks:
            callback.on_train_round_end()
