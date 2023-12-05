ADULT_RECORDINGS_PATH = "data/adult_vocalizations/"
FILE_NAME_PATTERN = "(.*)_(.*)[-_](.*)-(.*)\.wav"
SAMPLING_RATE = 22050
SAMPLE_LENGTH_MS = 300
TRAINING_EPOCHS = 1000

CALL_MAP = {
    'Ag' : ['Ag', 'AggC'],
    'Be' : ['Beggseq'],
    'DC' : ['DC', 'DisC'],
    'Di' : ['DI'],
    'LT' : ['LTC'],
    'Ne' : ['NestCSeq', 'NestC', 'NestSeq', 'NeKakleC', 'NekakleC', 'Ne', 'NestCseq', 'NeSeq', 'NeArkC', 'WhiCNestC', 'C'],
    'So' : ['Song', 'So'],
    'Te' : ['Tet', 'Te', 'TetC'],
    'Th' : ['ThuckC', 'TukC', 'ThuC', 'ThukC'],
    'Wh' : ['Whine', 'WhineC', 'WC', 'Whi', 'Wh', 'WhiC', 'WhineCSeq', 'WhiC', 'Wh']
}

CALL_IDS = {
    'Ag' : 0,
    'Be' : 1,
    'DC' : 2,
    'Di' : 3,
    'LT' : 4,
    'Ne' : 5,
    'So' : 6,
    'Te' : 7,
    'Th' : 8,
    'Wh' : 9
}