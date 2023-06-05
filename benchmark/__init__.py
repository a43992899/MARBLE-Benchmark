from benchmark.constants.model_constants import *
from benchmark.constants.task_constants import *

from benchmark.models.probers import *
from benchmark.models.musichubert_hf.hf_pretrains import *
from benchmark.tasks.GS.GS_prober import GSProber
from benchmark.tasks.EMO.EMO_prober import EMOProber
from benchmark.tasks.MAESTRO.MAESTRO_prober import MAESTROProber
from benchmark.tasks.MUSDB18.MUSDB18_prober import MUSDB18Prober
from benchmark.tasks.GTZAN.GTZANBT_prober import GTZANBTProber

MTTProber = ProberForBertUtterCLS
GTZANProber = ProberForBertUtterCLS
MTGGenreProber = ProberForBertUtterCLS
MTGInstrumentProber = ProberForBertUtterCLS
MTGMoodProber = ProberForBertUtterCLS
MTGTop50Prober = ProberForBertUtterCLS
NSynthIProber = ProberForBertUtterCLS
NSynthPProber = ProberForBertUtterCLS
VocalSetSProber = ProberForBertUtterCLS
VocalSetTProber = ProberForBertUtterCLS




