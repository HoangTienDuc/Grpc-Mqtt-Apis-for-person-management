import os

# Get Application Config
# import config

# Setup For Logging Init
import yaml
import logging
import app.modules.log_utils.logger_util

# Pull in Logging Config
path = os.path.join('app', 'modules', 'log_utils', 'logger_config.yaml')
with open(path, 'r') as stream:
    try:
      logging_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
      pass

# Load Logging configs
logging.config.dictConfig(logging_config)


log_level = logging.DEBUG


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
  log.setLevel(log_level)
