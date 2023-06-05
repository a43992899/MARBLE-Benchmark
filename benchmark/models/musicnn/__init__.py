import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import benchmark.models.musicnn.configuration as config
import benchmark.models.musicnn.models as models
import benchmark.models.musicnn.extractor as extractor
