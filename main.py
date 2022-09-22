import logging

import hydra

from t2v.config.root import RootConfig
from t2v.runner import Runner


@hydra.main(config_path="config", config_name="default", version_base=None)
def _main(cfg: RootConfig):
    logging.basicConfig()
    logging.info(f"Initializing...")
    Runner(cfg).run()


_main()
