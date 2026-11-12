# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import soundfile as sf

from deskref.audio_metrics import (
    get_path, load, frame_rms, floor_rms, event_mask, run_metrics,
    noise_reduction_db, lvl_change_db, sdr_db, mix_improvement,
)
from deskref.dsp import Processor

# %%
