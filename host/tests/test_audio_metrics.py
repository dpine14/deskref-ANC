import numpy as np
from deskref.dsp import Processor
import deskref.audio_metrics as am

class Half:
    def process_audio(self, mic, ref=None):
        return 0.5 * mic

#frame_rms() tests
def test_run_writes_each_block_in_place():
    x = np.random.default_rng(0).normal(0, 0.1, 5000).astype(np.float32)
    y = am.run_metrics(Half(), x, block=1024)
    assert np.allclose(y, 0.5 * x)

def test_frame_rms_hand_example():
    a = np.array([1, 1, 1, 1, 3, 3, 3, 3, 9], dtype=np.float32)
    assert np.allclose(am.frame_rms(a, 4), [1.0, 3.0])   # leftover 9 is dropped


def test_frame_rms_constant_signal():
    a = np.full(1000, 0.5, dtype=np.float32)
    assert np.allclose(am.frame_rms(a, 100), 0.5)


def test_frame_rms_sine_is_amplitude_over_root_two():
    t = np.arange(4800) / 48000
    a = np.sin(2 * np.pi * 1000 * t).astype(np.float32)   # 1 kHz, amplitude 1.0
    assert np.allclose(am.frame_rms(a, 480), 1 / np.sqrt(2), atol=1e-3)


def test_frame_rms_shorter_than_one_frame_is_empty():
    assert len(am.frame_rms(np.ones(10, dtype=np.float32), 480)) == 0


#rms_floor() tests
def test_floor_ignores_a_single_burst():
    s = np.full(48000, 0.001, dtype=np.float32)
    s[10000:10480] = 0.5                      # one loud burst, touches 2 of 100 frames
    assert np.isclose(am.floor_rms(s, 48000), 0.001)


def test_floor_of_constant_signal():
    s = np.full(48000, 0.25, dtype=np.float32)
    assert np.isclose(am.floor_rms(s, 48000), 0.25)


def test_floor_uses_requested_frame_length():
    s = np.full(48000, 0.1, dtype=np.float32)
    assert np.isclose(am.floor_rms(s, 48000, frame_ms=20), 0.1)

#event_mask() tests
RATE = 48000
def test_event_mask_finds_burst_only():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.001, RATE).astype(np.float32)        # 1 second of quiet noise
    start = int(0.5 * RATE)
    length = int(0.02 * RATE)                                  # 20 ms burst
    x[start : start + length] += rng.normal(0, 0.5, length)

    mask = am.event_mask(x, RATE, floor=0.001)

    assert len(mask) == len(x)
    assert mask[start + length // 2]       # inside the burst
    assert not mask[1000]                  # well away from it

def test_event_mask_one_sample_frames():
    x = np.array([1, 1, 1, 1, 10, 10, 1, 1, 1], dtype=np.float32)

    m = am.event_mask(x, 1000, floor=1.0, frame_ms=1, margin_db=15, pad=0)
    assert m.tolist() == [False, False, False, False, True, True, False, False, False]

    m = am.event_mask(x, 1000, floor=1.0, frame_ms=1, margin_db=15, pad=1)
    assert m.tolist() == [False, False, False, True, True, True, True, False, False]

def test_event_mask_frame_granularity():
    x = np.array([1, 1, 1, 1, 10, 10, 1, 1, 1], dtype=np.float32)
    m = am.event_mask(x, 1000, floor=1.0, frame_ms=3, margin_db=15, pad=0)
    assert m.tolist() == [False, False, False, True, True, True, False, False, False]

def test_event_mask_leftover_sample_stays_false():
    x = np.array([1, 1, 1, 1, 10, 10, 1, 1, 1], dtype=np.float32)
    m = am.event_mask(x, 1000, floor=1.0, frame_ms=4, margin_db=15, pad=0)
    assert len(m) == 9
    assert m.tolist() == [False, False, False, False, True, True, True, True, False]

def test_run_passthrough_preserves_signal():
    x = np.random.default_rng(4).normal(0, 0.1, 5000).astype(np.float32)
    assert np.array_equal(am.run_metrics(Processor(RATE), x, block=1024), x)

#Tests for mix_improvement()
class Passthrough:
    def process_audio(self, mic, ref=None):
        return mic


def test_mix_improvement_passthrough_is_zero():
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 0.1, 5000).astype(np.float32)
    noise = rng.normal(0, 0.05, 5000).astype(np.float32)
    before, after = am.mix_improvement(Passthrough, speech, noise, 1024)
    assert np.isclose(before, after, atol=0.5)

def test_mix_improvement_trims_to_shorter_length():
    speech = np.ones(1000, dtype=np.float32)
    noise = np.ones(600, dtype=np.float32) * 0.1
    before, after = am.mix_improvement(Passthrough, speech, noise, 256)
    assert np.isfinite(before) and np.isfinite(after)


class ReturnsSpeechOnly:
    """Cheating processor: knows the clean speech. Only useful as a test stand-in."""
    def __init__(self, speech):
        self.speech = speech
        self.pos = 0
    def process_audio(self, mic, ref=None):
        out = self.speech[self.pos : self.pos + len(mic)]
        self.pos += len(mic)
        return out


def test_mix_improvement_perfect_removal_is_large():
    rng = np.random.default_rng(1)
    speech = rng.normal(0, 0.1, 5000).astype(np.float32)
    noise = rng.normal(0, 0.05, 5000).astype(np.float32)
    before, after = am.mix_improvement(lambda: ReturnsSpeechOnly(speech), speech, noise, 1024)
    assert after > 100
