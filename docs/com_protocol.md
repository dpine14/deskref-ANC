Field			Size		Notes



Sync	                2 bytes	        0xA5 0x5A
Version	                1 byte	        1
Sequence	        uint16 LE	wraps at 65536
Timestamp	        uint32 LE	microseconds at the first sample, wraps
Sample count N	        uint16 LE	max 512
Samples	                N × int16 LE	raw ADC values
CRC	                uint16 LE	CRC-16/CCITT-FALSE over version through samples

