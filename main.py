import logging
import os
import sys

import hydra

from t2v.config.root import RootConfig
from t2v.runner import Runner

@hydra.main(config_path="config", config_name="default", version_base=None)
def _main(cfg: RootConfig):
    level = logging.getLevelName(os.getenv("LOG_LEVEL", default="INFO"))
    logging.basicConfig(level=level)
    logging.info(f"Initializing...")
    Runner(cfg).run()


_main()
