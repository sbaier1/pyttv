import typing

from omegaconf import DictConfig

from t2v.config.root import RootConfig


class InputVariableMechanism:
    """
    An InputVariableMechanism is an arbitrary form of multi-modal input for inducing some change in the animation.
    For example, audio beat detection will read a file and add a beat variable to the function context.
    """
    def __init__(self, config: DictConfig, root_config: RootConfig):
        pass

    def func_var_callback(self, t) -> typing.Dict[str, object]:
        """
        The function var callback is invoked by the function util to retrieve
        the current set of parameters for modulating functions.
        :return: the set of parameters as a dict that maps the variable name to the value. can include functions
        """
        return {}

    def prompt_modulator_callback(self, t) -> typing.Dict[str, object]:
        """
        The prompt modulator callback allows a mechanism to return text that can be used
        to modulate a prompt during a scene
        :return: the set of parameters as a dict that maps the variable name to the value. can include functions
        """
        return {}
