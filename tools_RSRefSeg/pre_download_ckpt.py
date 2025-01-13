import sys
import torch
sys.path.append(sys.path[0] + '/..')
import mmengine
import os
os.environ["MODELSCOPE_CACHE"] = "model_cache"
from mmseg.utils import register_all_modules
import rsris
from mmseg.registry import MODELS
register_all_modules()

config_file = 'configs_RSRefSeg/RSRefSeg-b.py'
config = mmengine.Config.fromfile(config_file)
model = MODELS.build(config.model)

config_file = 'configs_RSRefSeg/RSRefSeg-l.py'
config = mmengine.Config.fromfile(config_file)
model = MODELS.build(config.model)
