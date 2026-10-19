import struct
from dataclasses import dataclass

import numpy as np

SYNC = b"\xA5\x5A"
VERSION = 1
HEADER = struct.Struct("<BHIH")  # version, seq, timestamp_us, n_samples
CRC = struct.Struct("<H")
MAX_SAMPLES = 512


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_packet(seq: int, timestamp_us: int, samples) -> bytes:
    samples = np.asarray(samples, dtype=np.int16)
    body = HEADER.pack(VERSION, seq & 0xFFFF, timestamp_us & 0xFFFFFFFF, len(samples))
    body += samples.tobytes()
    return SYNC + body + CRC.pack(crc16_ccitt(body))


@dataclass
class Packet:
    seq: int
    timestamp_us: int
    samples: np.ndarray


class PacketParser:
    """Streaming parser: feed() raw bytes in any chunking, get complete packets back."""

    def __init__(self):
        self._buf = bytearray()
        self.packets_ok = 0
        self.crc_errors = 0
        self.dropped = 0
        self._last_seq = None

    def feed(self, data: bytes) -> list[Packet]:
        self._buf += data
        out = []
        while True:
            i = self._buf.find(SYNC)
            if i < 0:
                del self._buf[:-1]  # keep one byte in case it is half a sync word
                break
            del self._buf[:i]
            if len(self._buf) < 2 + HEADER.size:
                break
            ver, seq, ts, n = HEADER.unpack_from(self._buf, 2)
            if ver != VERSION or n > MAX_SAMPLES:
                del self._buf[:1]  # false sync, keep scanning
                continue
            total = 2 + HEADER.size + 2 * n + CRC.size
            if len(self._buf) < total:
                break
            body = bytes(self._buf[2 : total - CRC.size])
            (rx_crc,) = CRC.unpack_from(self._buf, total - CRC.size)
            if crc16_ccitt(body) != rx_crc:
                self.crc_errors += 1
                del self._buf[:1]
                continue
            if self._last_seq is not None:
                self.dropped += (seq - self._last_seq - 1) & 0xFFFF
            self._last_seq = seq
            self.packets_ok += 1
            samples = np.frombuffer(body[HEADER.size :], dtype="<i2").copy()
            out.append(Packet(seq, ts, samples))
            del self._buf[:total]
        return out