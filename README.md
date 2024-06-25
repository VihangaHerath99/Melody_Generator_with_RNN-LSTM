# Generating melodies with RNN-LSTM
__Vihanga Herath__

> #### Python Version
> - Python 3.10.6
> #### Main Libraries
> - tensorflow -- 2.16.1
> - music21 -- 9.1.0
> #### Tools
> - MuseScore [[Link]](https://musescore.org/en)
> #### DataSet
> - Kern Scores Dataset [[Link]](https://kern.humdrum.org/cgi-bin/browse?l=essen%2Feuropa%2Fdeutschl)
> #### To paly with this code
> - download this repo
> - create a virtual environment and install the dependencies using __requirements.txt__
> - download the __MuseScore__ using the given link
> - Set up environment variables with music21 [[Check this Link]](https://web.mit.edu/music21/doc/moduleReference/moduleEnvironment.html)
> - Now you can generate melodies using [melodyGenerator.py](melodyGenerator.py)


## Basic Concepts

### Melody
Sequence of notes and rests
![image_001](<images/01_melody.png>)

### Pitch
Indicates how high/low a note is
![image_002](<images/02_pitch.png>)

### Scientific pitch notation
- Note name + octave
- Eg:- C3, D4, A1
![image_003](<images/03_scientific_pitch_notation.png>)

### MIDI note notation
- MIDI is a protocol to play, edit and record music
- Map note names to numbers
- Eg:- C4 = 60
![image_004](<images/04_MIDI_note_notation.png>)

## Melody Generation Problem
- Treat melody as a time series
- Time-series prediction problem
- MIDI note notation is used

## Preprocessing Steps
1. Load the song files in the dataset.
2. Filter out songs that have non-acceptable durations.
3. Transpose songs to C-major / A-minor.
4. Encode songs with midi notation.
5. Save songs to text files.
6. Create one text file that contains all songs.
7. Create the vocabulary of symbols.
8. Map song symbols to integers using the vocabulary.
9. Generate training sequences and the targets.

See [preprocess.py](preprocess.py) for the source code.

##  Model Architecture
<div style="display: flex; justify-content: space-around;">
    <img src="images\05_model_architecture.png" alt="05_model_architecture" width="200"/>
    <img src="images\06_model_summary.png" alt="06_model_summary" width="500"/>
</div>

## Model Training
The model is trained for 50 epochs using the 'sparse_categorical_crossentropy' loss function and the 'Adam optimizer'.  
See [train.py](train.py) for the source code.

<!-- ![image_007](<images\07_loss curve.png>) -->

<img src="images\07_loss curve.png" alt="image_007" width="600"/>



## Generating Melodies

Sampling is done using temperatures to adjust the model's output probabilities:

> predictions = np.log(probabilities) / temperature    
>probabilities = np.exp(predictions) / np.sum(np.exp(predictions))  

When the temperature is close to 0, the model's output is more deterministic. This means that the model will favor the most likely predictions strongly, leading to less variation in the generated melodies.

Conversely, when the temperature is close to 1.0, the model's output has more randomness. This means the differences between the predicted probabilities are reduced, allowing for more diverse and less predictable melodies.  

See the [melodyGenerator.py](melodyGenerator.py) for the source code.

You can find the generated melodies in the [generated melodies](generated%20melodies) folder.
