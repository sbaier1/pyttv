"""
Note: this may contain errors in the lower octaves (-1, -2). extracted/parsed from https://studiocode.dev/resources/midi-middle-c/
"""

name_to_note = {
    "G9": 127,
    "F#9": 126,
    "Gb9": 126,
    "F9": 125,
    "E9": 124,
    "D#9": 123,
    "Eb9": 123,
    "D9": 122,
    "C#9": 121,
    "Db9": 121,
    "C9": 120,
    "B8": 119,
    "A#8": 118,
    "Bb8": 118,
    "A8": 117,
    "G#8": 116,
    "Ab8": 116,
    "G8": 115,
    "F#8": 114,
    "Gb8": 114,
    "F8": 113,
    "E8": 112,
    "D#8": 111,
    "Eb8": 111,
    "D8": 110,
    "C#8": 109,
    "Db8": 109,
    "C8": 108,
    "B7": 107,
    "A#7": 106,
    "Bb7": 106,
    "A7": 105,
    "G#7": 104,
    "Ab7": 104,
    "G7": 103,
    "F#7": 102,
    "Gb7": 102,
    "F7": 101,
    "E7": 100,
    "D#7": 99,
    "Eb7": 99,
    "D7": 98,
    "C#7": 97,
    "Db7": 97,
    "C7": 96,
    "B6": 95,
    "A#6": 94,
    "Bb6": 94,
    "A6": 93,
    "G#6": 92,
    "Ab6": 92,
    "G6": 91,
    "F#6": 90,
    "Gb6": 90,
    "F6": 89,
    "E6": 88,
    "D#6": 87,
    "Eb6": 87,
    "D6": 86,
    "C#6": 85,
    "Db6": 85,
    "C6": 84,
    "B5": 83,
    "A#5": 82,
    "Bb5": 82,
    "A5": 81,
    "G#5": 80,
    "Ab5": 80,
    "G5": 79,
    "F#5": 78,
    "Gb5": 78,
    "F5": 77,
    "E5": 76,
    "D#5": 75,
    "Eb5": 75,
    "D5": 74,
    "C#5": 73,
    "Db5": 73,
    "C5": 72,
    "B4": 71,
    "A#4": 70,
    "Bb4": 70,
    "A4": 69,
    "G#4": 68,
    "Ab4": 68,
    "G4": 67,
    "F#4": 66,
    "Gb4": 66,
    "F4": 65,
    "E4": 64,
    "D#4": 63,
    "Eb4": 63,
    "D4": 62,
    "C#4": 61,
    "Db4": 61,
    "C4": 60,
    "B3": 59,
    "A#3": 58,
    "Bb3": 58,
    "A3": 57,
    "G#3": 56,
    "Ab3": 56,
    "G3": 55,
    "F#3": 54,
    "Gb3": 54,
    "F3": 53,
    "E3": 52,
    "D#3": 51,
    "Eb3": 51,
    "D3": 50,
    "C#3": 49,
    "Db3": 49,
    "C3": 48,
    "B2": 47,
    "A#2": 46,
    "Bb2": 46,
    "A2": 45,
    "G#2": 44,
    "Ab2": 44,
    "G2": 43,
    "F#2": 42,
    "Gb2": 42,
    "F2": 41,
    "E2": 40,
    "D#2": 39,
    "Eb2": 39,
    "D2": 38,
    "C#2": 37,
    "Db2": 37,
    "C2": 36,
    "B1": 35,
    "A#1": 34,
    "Bb1": 34,
    "A1": 33,
    "G#1": 32,
    "Ab1": 32,
    "G1": 31,
    "F#1": 30,
    "Gb1": 30,
    "F1": 29,
    "E1": 28,
    "D#1": 27,
    "Eb1": 27,
    "D1": 26,
    "C#1": 25,
    "Db1": 25,
    "C1": 24,
    "B0": 23,
    "A#0": 22,
    "Bb0": 22,
    "A0": 21,
    "G#-1": 20,
    "G-1": 19,
    "F#-1": 18,
    "F-1": 17,
    "E-1": 16,
    "D#-1": 15,
    "D-1": 14,
    "C#-1": 13,
    "C0-1": 12,
    "B-1": 11,
    "A#-1": 10,
    "A-1": 9,
    "G#-2": 8,
    "G-2": 7,
    "F#-2": 6,
    "F-2": 5,
    "E-2": 4,
    "D#-2": 3,
    "D-2": 2,
    "C#-2": 1,
    "C-1": 0,
}

name_to_note_no_oct = {'G': [127, 115, 103, 91, 79, 67, 55, 43, 31, 19, 7],
                       'F#': [126, 114, 102, 90, 78, 66, 54, 42, 30, 18, 6],
                       'Gb': [126, 114, 102, 90, 78, 66, 54, 42, 30],
                       'F': [125, 113, 101, 89, 77, 65, 53, 41, 29, 17, 5],
                       'E': [124, 112, 100, 88, 76, 64, 52, 40, 28, 16, 4],
                       'D#': [123, 111, 99, 87, 75, 63, 51, 39, 27, 15, 3],
                       'Eb': [123, 111, 99, 87, 75, 63, 51, 39, 27],
                       'D': [122, 110, 98, 86, 74, 62, 50, 38, 26, 14, 2],
                       'C#': [121, 109, 97, 85, 73, 61, 49, 37, 25, 13, 1],
                       'Db': [121, 109, 97, 85, 73, 61, 49, 37, 25],
                       'C': [120, 108, 96, 84, 72, 60, 48, 36, 24, 12, 0],
                       'B': [119, 107, 95, 83, 71, 59, 47, 35, 23, 11],
                       'A#': [118, 106, 94, 82, 70, 58, 46, 34, 22, 10],
                       'Bb': [118, 106, 94, 82, 70, 58, 46, 34, 22],
                       'A': [117, 105, 93, 81, 69, 57, 45, 33, 21, 9],
                       'G#': [116, 104, 92, 80, 68, 56, 44, 32, 20, 8],
                       'Ab': [116, 104, 92, 80, 68, 56, 44, 32]
                       }

note_to_name = {
    127: "G9",
    126: "F#9/Gb9",
    125: "F9",
    124: "E9",
    123: "D#9/Eb9",
    122: "D9",
    121: "C#9/Db9",
    120: "C9",
    119: "B8",
    118: "A#8/Bb8",
    117: "A8",
    116: "G#8/Ab8",
    115: "G8",
    114: "F#8/Gb8",
    113: "F8",
    112: "E8",
    111: "D#8/Eb8",
    110: "D8",
    109: "C#8/Db8",
    108: "C8",
    107: "B7",
    106: "A#7/Bb7",
    105: "A7",
    104: "G#7/Ab7",
    103: "G7",
    102: "F#7/Gb7",
    101: "F7",
    100: "E7",
    99: "D#7/Eb7",
    98: "D7",
    97: "C#7/Db7",
    96: "C7",
    95: "B6",
    94: "A#6/Bb6",
    93: "A6",
    92: "G#6/Ab6",
    91: "G6",
    90: "F#6/Gb6",
    89: "F6",
    88: "E6",
    87: "D#6/Eb6",
    86: "D6",
    85: "C#6/Db6",
    84: "C6",
    83: "B5",
    82: "A#5/Bb5",
    81: "A5",
    80: "G#5/Ab5",
    79: "G5",
    78: "F#5/Gb5",
    77: "F5",
    76: "E5",
    75: "D#5/Eb5",
    74: "D5",
    73: "C#5/Db5",
    72: "C5",
    71: "B4",
    70: "A#4/Bb4",
    69: "A4",
    68: "G#4/Ab4",
    67: "G4",
    66: "F#4/Gb4",
    65: "F4",
    64: "E4",
    63: "D#4/Eb4",
    62: "D4",
    61: "C#4/Db4",
    60: "C4",
    59: "B3",
    58: "A#3/Bb3",
    57: "A3",
    56: "G#3/Ab3",
    55: "G3",
    54: "F#3/Gb3",
    53: "F3",
    52: "E3",
    51: "D#3/Eb3",
    50: "D3",
    49: "C#3/Db3",
    48: "C3",
    47: "B2",
    46: "A#2/Bb2",
    45: "A2",
    44: "G#2/Ab2",
    43: "G2",
    42: "F#2/Gb2",
    41: "F2",
    40: "E2",
    39: "D#2/Eb2",
    38: "D2",
    37: "C#2/Db2",
    36: "C2",
    35: "B1",
    34: "A#1/Bb1",
    33: "A1",
    32: "G#1/Ab1",
    31: "G1",
    30: "F#1/Gb1",
    29: "F1",
    28: "E1",
    27: "D#1/Eb1",
    26: "D1",
    25: "C#1/Db1",
    24: "C1",
    23: "B0",
    22: "A#0/Bb0",
    21: "A0",
    20: "G#-1",
    19: "G-1",
    18: "F#-1",
    17: "F-1",
    16: "E-1",
    15: "D#-1",
    14: "D-1",
    13: "C#-1",
    12: "C-1",
    11: "B-2",
    10: "A#-2",
    9: "A-2",
    8: "G#-2",
    7: "G-2",
    6: "F#-2",
    5: "F-2",
    4: "E-2",
    3: "D#-2",
    2: "D-2",
    1: "C#-2",
    0: "C-2",
}
