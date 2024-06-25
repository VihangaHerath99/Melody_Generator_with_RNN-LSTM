import os
import music21 as m21
import json
from tensorflow import keras
import numpy as np
from tqdm import tqdm


KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset.txt"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64


def load_songs_in_kern(dataset_path):
    '''
    Load the all krn files in the given path

    Args:
        dataset_path (string) -- path to dataset

    Returns:
        songs (list) -- list of loaded songs
    '''
    songs = []
    print('Loading songs....')
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    print(f"{len(songs)} songs are loaded")
    return songs


def has_acceptable_duarations(song, acceptable_durations):
    '''
    Check whether the song has acceptable durations
    
    Args:
        song (music21.stream.base.Score) -- song file to check duration
        acceptable_durations (list) -- list of acceptable duration times in refference quarter note
    
    Returns:
        True/False (boolen) -- status of the duration check
    '''
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    '''
    Transpose the song to C-major or A-minor
    
    Args:
        song (music21.stream.base.Score) -- song file to transpose

    Returns:
        transposed_song (music21.stream.base.Score) -- transposed song file
    '''
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    # get interval for transposition E.g. Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song, time_step=0.25):
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]
    '''encode the song using midi notation
    
    Args:
        song (music21.stream.base.Score) -- song file to encode
        time_step (float or integer) -- time gap between two symbols in reference to quarterlength

    Returns:
        encoded_song (string) -- string with encoded symbols
    '''
    encoded_song = []
    for event in song.flat.notesAndRests:
        #handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def preprocess(dataset_path):
    '''
    Prepocess the krn files and generate encoded song files.

    Args:
        dataset_path (string) -- path to the raw dataset
    '''
    # load the folk songs
    songs = load_songs_in_kern(dataset_path)
    for i, song in enumerate(tqdm(songs, desc="Processing Songs", unit="song")):
        # filter out songs that hav e non-acceptable durations
        if not has_acceptable_duarations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to C-major/A-minor
        song = transpose(song)
        # encode songs with music time series representation
        encoded_song = encode_song(song)
        # save songs to text
        save_path = os.path.join(SAVE_DIR, str(i)+'.txt')
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    '''
    Read encoded song file.

    Args:
        file_path (string) -- path to text file that contains encoded song

    Returns:
        song (string) -- symbols of the encoded song
    '''
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    '''
    Create a single text file that contains all encoded songs.

    Args:
        dataset_path (string) -- path to encoded song dataset
        file_dataset_path (string) -- path to save the text file that contains all encoded songs
        sequence_length (integer) -- length of a sequence

    Returns:
        songs (string) -- text file that contains all encoded songs
    '''
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]
    # save string that contains all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    '''
    create mappings for the encoded songs

    Args:
        songs (string) -- text file that contains all encoded songs
        mapping_path (string) -- path to save json file with mapping
    '''
    mappings = {}
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    # create mapping
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    # save vocabularyto a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def convert_songs_to_int(songs):
    '''
    convert symbols in songs to integer list

    Args:
        songs (string) -- text file that contains all encoded songs

    Returns:
        int_songs (list) -- mapped integers for symbols in songs 
    '''
    int_songs = []
    # load the mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    # cast songs string to a list
    songs = songs.split()
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs

def generate_training_sequences(sequence_length):
    '''
    Generate training sequences and targets.

    Args:
        sequence_length (integer) -- length of a sequence

    Returns:
        inputs (numpy ndarray) -- training input sequences
        targets (numpy ndarray) -- training targets
    '''
    # load songs and map them into int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    # generate the training sequences
    # 100 symbols, 64 sl, 100-64 = 36 sequences
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)
    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()