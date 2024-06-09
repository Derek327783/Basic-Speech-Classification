# Speech Recognition Deployment Basics

## 1. Dataset
- The dataset is sourced from Google's Speech-Command dataset which comprises of 65,000 one second long audio wav files that speak 30 different kinds of words. More information can be found in this [link](https://research.google/blog/launching-the-speech-commands-dataset/).

## 2. Feature Extraction
- The feature extraction portion uses the Mel-Frequency Cepstral Coefficients(MFCCs) to extract and condense information of various sections of the audio files to be fed into the deep learning model.

- Below is the code to extract the MFCCs.
```
file_path = os.path.join(dirpath,audio_files) # Get file path
                file_path = file_path.replace("\\", "/")
                signal,sr = librosa.load(file_path, sr = 22050) # Load audio file
                if len(signal) >= SAMPLES_TO_CONS: # Check audio is at least 1 sec long
                    signal = signal[:SAMPLES_TO_CONS] # Make sure audio is exactly 1 sec long for consistency to fit into NN
                    MFCC = librosa.feature.mfcc(y=signal,n_mfcc=n_mfcc, hop_length = hop_length, n_fft = n_fft)
```
- More information can be found in  **pythonProject/classifier/prepare_dataset.py file**

## 3. Model Architecture
- The model architecture utilises the Convolutional Neural Network layer to take in the MFCC feature matrix of each audio file and returns a vector of probabilities. 

- Below is the architecture of the model

```
model = keras.Sequential()

    # convo layer 1
    model.add(keras.layers.Conv2D(64,(3,3),activation = "relu",
                                  input_shape = input_shape,
                                  kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))

    # convo layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # convo layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # use softmax to represent as probabilities
    model.add(keras.layers.Dense(NUM_KEYWORDS,activation = "softmax"))

    # Compile model, like consolidating the model with all its parameters
    optimiser = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
    model.compile(optimizer=optimiser,loss=error, metrics = ["accuracy"])
```
- More information can be found in **pythonProject/classifier.train.py**


