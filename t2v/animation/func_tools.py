import logging
import math

import numpy as np

from t2v.config.root import RootConfig


class FuncUtil:
    def __init__(self, cfg: RootConfig):
        self.math_env = {
            "abs": abs,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "np": np,
            "__builtins__": None,
        }
        self.cfg = cfg
        self.math_env.update(
            {key: getattr(math, key) for key in dir(math) if "_" not in key}
        )
        self.eval_memo = {}
        self.callbacks = {}
        self.prev_values = {}

    def parametric_eval(self, string, t, **vals):
        if isinstance(string, str):
            memo_key = f"{string}{t}"
            if memo_key in self.eval_memo:
                return self.eval_memo[memo_key]
            self.math_env.update(vals)
            self.update_math_env(t)
            self.update_funcs(t)
            output = self.actual_eval(string, memo_key)
            return output
        else:
            return string

    def actual_eval(self, string, memo_key=None):
        try:
            output = eval(string, self.math_env)
        except SyntaxError as e:
            raise RuntimeError("Error in parametric value " + string)
        except TypeError as e:
            raise RuntimeError(
                f"Could not evaluate string {string}, missing variable? variables in context: {self.math_env}")
        if memo_key is not None:
            self.eval_memo[memo_key] = output
        return output

    def update_funcs(self, t):
        if self.cfg.additional_context.custom_functions is not None:
            for func in self.cfg.additional_context.custom_functions:
                value = self.actual_eval(func.function)
                self.math_env[func.variable_name] = value
                if func.prev_values > 0:
                    if func.variable_name not in self.prev_values:
                        self.prev_values[func.variable_name] = []
                    prev_values = self.prev_values[func.variable_name]
                    prev_values.append(value)
                    if len(prev_values) > func.prev_values:
                        del(prev_values[0])

    def update_math_env(self, t):
        self.math_env["t"] = t
        for callback in self.callbacks.keys():
            callback_result = self.callbacks[callback](t)
            logging.debug(f"func callback invocation result for {callback}: {callback_result}")
            self.math_env.update(callback_result)
        return self.math_env

    def add_callback(self, key, func):
        """
        Adds a callback function. When parametric_eval is called,
        these callbacks may add additional context (variables) to the evaluated function
        :param key: string key of the callback for identification and if necessary later removal
        :param func: function to add. must take one argument (t, time in seconds as float) and return a typing.Dict[str, float]
        """
        self.callbacks[key] = func
