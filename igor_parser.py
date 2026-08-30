"""
Igor Pro .ibw file parser for AFM force curve data
Extracts deflection, Z position, and metadata from Igor binary wave files
"""

import struct
import numpy as np
from typing import Dict, Tuple, Optional


class IgorParser:
    """Parse Igor Pro .ibw binary wave files"""

    def __init__(self, filepath: str):
        """Initialize parser with filepath"""
        self.filepath = filepath
        self.data = None
        self.metadata = {}

    def parse(self) -> Dict:
        """
        Parse Igor file and extract wave data

        Returns:
        --------
        dict with 'data' (deflection values) and 'metadata'
        """
        try:
            with open(self.filepath, 'rb') as f:
                # Read version info
                version = struct.unpack('<I', f.read(4))[0]

                # Skip to wave header (simplified parsing)
                # Igor .ibw files have complex structure, so we extract key info
                f.seek(0)
                content = f.read()

                # Extract wave name
                wave_name = self._extract_wave_name(content)

                # Extract numeric data (floats at end of file)
                wave_data = self._extract_wave_data(content)

                self.metadata = {
                    'filename': self.filepath,
                    'wave_name': wave_name,
                    'data_points': len(wave_data) if wave_data is not None else 0
                }

                self.data = wave_data

                return {
                    'data': wave_data,
                    'metadata': self.metadata
                }

        except Exception as e:
            print(f"Error parsing Igor file: {e}")
            return {'data': None, 'metadata': {'error': str(e)}}

    def _extract_wave_name(self, content: bytes) -> str:
        """Extract wave name from Igor file"""
        try:
            # Look for text pattern "Sample" followed by numbers
            text = content.decode('latin-1', errors='ignore')

            # Find wave name (usually after header)
            if 'Sample' in text:
                idx = text.find('Sample')
                name_part = text[idx:idx+20]
                # Clean up null bytes and whitespace
                name = name_part.split('\x00')[0]
                return name

            return 'Unknown'
        except:
            return 'Unknown'

    def _extract_wave_data(self, content: bytes) -> Optional[np.ndarray]:
        """
        Extract numeric wave data from Igor file
        Igor stores floating point data, usually as single or double precision
        """
        try:
            # Igor files store data as doubles (8 bytes) or floats (4 bytes)
            # Try to extract as double precision floats

            # Find where numeric data starts (after header, usually around offset 500+)
            # Look for blocks of valid float data

            # Simplified approach: scan through file looking for float-like data
            for offset in range(500, len(content) - 8, 4):
                try:
                    # Try reading 4 bytes as float
                    val = struct.unpack('<f', content[offset:offset+4])[0]

                    # Check if value is reasonable (not NaN or huge)
                    if not np.isnan(val) and abs(val) < 1e10:
                        # Found likely start of numeric data
                        # Extract remaining floats
                        num_points = (len(content) - offset) // 4

                        data = struct.unpack(f'<{num_points}f', content[offset:offset+num_points*4])
                        return np.array(data)
                except:
                    continue

            # If float parsing fails, try double precision
            for offset in range(500, len(content) - 8, 8):
                try:
                    val = struct.unpack('<d', content[offset:offset+8])[0]
                    if not np.isnan(val) and abs(val) < 1e10:
                        num_points = (len(content) - offset) // 8
                        data = struct.unpack(f'<{num_points}d', content[offset:offset+num_points*8])
                        return np.array(data)
                except:
                    continue

            return None

        except Exception as e:
            print(f"Error extracting wave data: {e}")
            return None


def load_igor_pair(filepath1: str, filepath2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pair of Igor files (approach and retract curves)

    Parameters:
    -----------
    filepath1, filepath2 : str
        Paths to two Igor .ibw files

    Returns:
    --------
    (data1, data2) : tuple of numpy arrays
    """
    parser1 = IgorParser(filepath1)
    parser2 = IgorParser(filepath2)

    result1 = parser1.parse()
    result2 = parser2.parse()

    return result1['data'], result2['data']
