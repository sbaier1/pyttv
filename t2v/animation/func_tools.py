import logging
import math

import numpy as np


class FuncUtil:
    def __init__(self):
        self.math_env = {
            "abs": abs,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "np": np,
            "__builtins__": None,
        }
        self.math_env.update(
            {key: getattr(math, key) for key in dir(math) if "_" not in key}
        )
        self.eval_memo = {}
        self.callbacks = {}

    def parametric_eval(self, string, t, **vals):
        # if string in self.eval_memo:
        #    return self.eval_memo[string]
        if isinstance(string, str):
            self.math_env.update(vals)
            self.math_env["t"] = t
            for callback in self.callbacks.keys():
                callback_result = self.callbacks[callback](t)
                logging.info(f"func callback invocation result for {callback}: {callback_result}")
                self.math_env.update(callback_result)
            try:
                output = eval(string, self.math_env)
            except SyntaxError as e:
                raise RuntimeError("Error in parametric value " + string)
            self.eval_memo[string] = output
            return output
        else:
            return string

    def add_callback(self, key, func):
        """
        Adds a callback function. When parametric_eval is called,
        these callbacks may add additional context (variables) to the evaluated function
        :param key: string key of the callback for identification and if necessary later removal
        :param func: function to add. must take one argument (t, time in seconds as float) and return a typing.Dict[str, float]
        """
        self.callbacks[key] = func
