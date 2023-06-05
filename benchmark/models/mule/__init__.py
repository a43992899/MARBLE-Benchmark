import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from benchmark.models.mule import configuration as config
from benchmark.models.mule import mule as mule
