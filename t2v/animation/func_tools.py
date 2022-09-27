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

    def parametric_eval(self, string, t, **vals):
        #if string in self.eval_memo:
        #    return self.eval_memo[string]
        if isinstance(string, str):
            self.math_env.update(vals)
            self.math_env["t"] = t
            try:
                output = eval(string, self.math_env)
            except SyntaxError as e:
                raise RuntimeError("Error in parametric value " + string)
            self.eval_memo[string] = output
            return output
        else:
            return string

