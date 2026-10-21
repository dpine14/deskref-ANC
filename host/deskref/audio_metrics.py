import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from deskref.dsp import Processor

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EPS = 1e-12
        
#Load audio file and use 1st channel if stereo
def load(path)-> tuple[np.ndarray, int]:

    x, rate = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]  # Use only the first channel
    return x, rate

#Rule of squares calculation for energy of the signal
def energy(x)-> float:
    return float(np.sum(np.square(x, dtype=np.float64)))

def db(x)-> float:
    return float(10*np.log10(max(x, EPS)))

def run_metrics(proc,x: np.ndarray,block: int= 1024)-> np.ndarray:        
    x_out = np.empty_like(x)
    for i in range(0, len(x), block):
        x_out[i:i+block] = proc.process_audio(x[i:i+block])
    return x_out
#
def main():
    x,rate = load("data/raw/bumps_02.wav")
    return("Fin")




if __name__ == "__main__":
    main()