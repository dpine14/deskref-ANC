import numpy as np

from deskref.com_protocol import PacketParser, crc16_ccitt, encode_packet


def test_crc_check_value():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_roundtrip():
    s = np.array([-5, 0, 1234], dtype=np.int16)
    pkts = PacketParser().feed(encode_packet(7, 123456, s))
    assert len(pkts) == 1
    assert pkts[0].seq == 7 and pkts[0].timestamp_us == 123456
    assert np.array_equal(pkts[0].samples, s)


def test_byte_by_byte():
    parser = PacketParser()
    stream = encode_packet(0, 0, [1, 2, 3]) + encode_packet(1, 100, [4, 5])
    pkts = [p for b in stream for p in parser.feed(bytes([b]))]
    assert [p.seq for p in pkts] == [0, 1]


def test_garbage_before_sync():
    pkts = PacketParser().feed(b"\x00\xff\x13" + encode_packet(0, 0, [9]))
    assert len(pkts) == 1


def test_corrupt_packet_is_skipped_and_counted():
    bad = bytearray(encode_packet(0, 0, [1, 2, 3]))
    bad[-4] ^= 0xFF
    parser = PacketParser()
    pkts = parser.feed(bytes(bad) + encode_packet(1, 0, [4, 5]))
    assert [p.seq for p in pkts] == [1]
    assert parser.crc_errors == 1


def test_dropped_packets_counted():
    parser = PacketParser()
    parser.feed(encode_packet(0, 0, [1]) + encode_packet(3, 0, [1]))
    assert parser.dropped == 2