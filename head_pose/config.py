import os


# names:
DATA_PATH = './data/JET_data/processed_data/'

SPLIT_TABLE_PATH = './data/split_tables/'
SPLIT_TABLE_NAME = 'split_table.json'

DEBUG_FOLDER = './data/CV_debug/'

for f in [DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)
