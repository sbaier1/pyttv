import logging
import re
from datetime import timedelta

time_regex = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?((?P<milliseconds>\d+)?ms)?')


def parse_time(time_str):
    parts = time_regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)
    result = timedelta(**time_params)
    if result.total_seconds() == 0:
        logging.warning(f"Time string {time_str} evaluated to 0s. 0 durations typically don't need to be specified."
                        f" (Could be a parser error)")
    return result
