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

def run_metrics(
        proc, x: np.ndarray,
        block: int= 1024
        )-> np.ndarray:        
    x_out = np.empty_like(x)
    for i in range(0, len(x), block):
        x_out[i:i+block] = proc.process_audio(x[i:i+block])
    return x_out

#Compute the RMS for a given frame size
def frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    x = x[:n*frame].astype(np.float64).reshape(n, frame)

    return np.sqrt(np.mean(x**2, axis=1))

#Compute the median RMS of the soundfile, which is used to determine the noise floor of the recording
def floor_rms(silence: np.ndarray, rate: int,
        frame_ms: float=10.0
        ) -> float:
    if int(rate * frame_ms / 1000) <= 0:
        raise ValueError("Recording file is less than one frame long")
    return float(np.median(frame_rms(silence, int(rate * frame_ms / 1000))))

#Apply mask based on the RMS of the signal to determine which frames are considered "events" (i.e., above the noise floor)
def event_mask(x: np.ndarray, rate: int, 
        floor, frame_ms: float=10.0,
        margin_db: float=12.0, pad: int=2
        ) -> np.ndarray:
    #Need to find a better way to determine margin_db, as it is dependent on the recording environment and the noise floor of the recording
    frame = int(rate*frame_ms / 1000)
    hot = frame_rms(x, frame) > (floor * 10**(margin_db/20))
    if pad:
        hot = np.convolve(hot.astype(int), np.ones(2*pad+1, int), mode="same") > 0
    mask = np.zeros(len(x), dtype= bool)
    mask[:len(hot)*frame] = np.repeat(hot, frame)

    return mask

#return mask reduction in db
def noise_reduction_db(x, y, mask):
    if not mask.any():
        return float("nan") 
    return db(energy(x[mask])) - db(energy(y[mask]))

def lvl_change_db(x,y):
    return db(energy(y)) - db(energy(x))

def sdr_db(ref, y):
    return db(energy(ref)) - db(energy(y - ref))

def mix_improvement(make_proc, speech, noise, block):
    length = min(len(speech), len(noise))
    speech, noise = speech[:length], noise[:length]
    out = run_metrics(make_proc(), speech + noise, block)

    return db(energy(speech)) - db(energy(noise)), sdr_db(speech, out)

def get_path(name: str) -> Path:
    return RAW_DIR / f"{name}.wav"

def gate_warning(samples, rate, frame_ms: float=10.0):
    frame = max(1, int(rate*frame_ms / 1000))
    samples = frame_rms(samples, frame)
    return np.count_nonzero(samples == 0) / len(samples)

def noise_report(label, samples, rate, floor, block, margin_db):
    
    p = Processor(rate)
    out = run_metrics(p, samples, block)
    mask = event_mask(samples, rate, floor, margin_db=margin_db)
    nr = noise_reduction_db(samples, out, mask)
    print(f"{label:8s} noise reduction {nr:6.1f} dB  ({mask.mean() * 100:.1f}% flagged)")

def main():
    
    #Define aguments to be taken and parsed
    p = argparse.ArgumentParser()
    p.add_argument("--silence", default="silence_03")
    p.add_argument("--speaking", default="talking_01")
    p.add_argument("--bumps", default="bumps_01")
    p.add_argument("--typing", default="typing_02")
    p.add_argument("--margin", type=float, default=12.0)
    p.add_argument("--block", type=int, default=1024)
    args = p.parse_args()

    #Dict to easily choose which file
    files, samples, rates = {
        "silence": args.silence,
        "typing": args.typing,
        "speaking": args.speaking,
        "bumps": args.bumps,
    },{},{}

    
    #Run through and load audio files
    for label, name in files.items():
        #Check that file actually exists
        if not get_path(name).exists():
            raise SystemExit(f"The file {get_path(name)} in {get_path(name).parent} does not exist") 
        x, rate = load(get_path(name))

        #Check that recording isnt empty and then make sure that the file doesn't have clipping
        if len(x) == 0: raise SystemExit(f"{get_path(name)} is an empty file")
        if np.abs(x).max() > 0.99: 
            print(f"WARNING: {get_path(name).name} has clipping")
        samples[label] = x
        rates[label] = rate
        frac = gate_warning(x, rate)

        #Check if file has gating
        if  frac >= 0.02: 
            print(f"WARNING: {get_path(name).name} has {frac*100:.1f}% zero frames, possibly gated")

    #Now that weve loaded files, we check to make sure they have identical rates. Otherwise comparison math wont work correctly
    if len(set(rates.values())) != 1: 
        raise SystemExit(f"Sample rates are not identical: {rates}")

    rate = rates["silence"]
    floor = floor_rms(samples["silence"], rate)
    sp, r = load(get_path(args.speaking))
    ps = run_metrics(Processor(rate), sp, args.block)
    
    for label in ["typing", "bumps"]: 
        noise_report(label, samples[label], rate, floor, args.block, args.margin)

    x, rate = load(get_path("typing_03"))
    r = frame_rms(x, int(rate * 10 / 1000))
    d = lambda v: 20 * np.log10(max(v, 1e-9))    

    before, after = mix_improvement(lambda: Processor(rate), samples["speaking"], samples["typing"], args.block)
    print(f"{args.speaking}     Lvl Change: {lvl_change_db(sp, ps)}     SDR: {sdr_db(sp, ps)}") 
    print(d(floor), d(np.median(r)), d(np.percentile(r, 99)), d(r.max()))


    



if __name__ == "__main__":
    main()