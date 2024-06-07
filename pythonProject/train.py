import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 40
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_KEYWORDS = 4

def load_dataset(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)
    X = np.array(data["MFCCs"])
    Y = np.array(data["labels"])
    return X,Y
def get_data_splits(dataset_path, test_size = 0.1, validation_size = 0.1):
    # load dataset
    X,Y = load_dataset(dataset_path)

    # create train/validation/test splits
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,
                                                                 test_size=validation_size)

    # convert inputs from 2d to 3d for CNN
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    return X_train,X_validation,X_test,Y_train,Y_validation,Y_test

def build_model(input_shape,learning_rate,error = "sparse_categorical_crossentropy"):
    # Build Network
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

    # model overview
    model.summary()

    return model
def main():
    # Load train data splits
    X_train,X_validation,X_test,Y_train,Y_validation,Y_test = get_data_splits(DATASET_PATH)

    # Build CNN model
    input_shape = (X_train.shape[1],X_train.shape[2],1) # (n_fft * n_mfccs * channels)
    model = build_model(input_shape,learning_rate = LEARNING_RATE)

    # Train the model
    model.fit(X_train,Y_train,epochs=EPOCHS,batch_size = BATCH_SIZE, validation_data = (X_validation,Y_validation))

    # Evaluate/test the model
    test_error,test_accuracy = model.evaluate(X_test,Y_test)
    print(f"test_error: {test_error}\n test_accuracy: {test_accuracy}")

    # Save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()
# test_size = 0.1
# validation_size = 0.1
# X,Y = load_dataset(DATASET_PATH)
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size)
# X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,
#                                                                 test_size=validation_size)
# input_shape = (X_train.shape[1],X_train.shape[2],1)
# model = build_model(input_shape=input_shape,learning_rate=LEARNING_RATE)
#
# print(X_train.shape)
# print(Y_train.shape)
# print(X_validation.shape)
# print(Y_validation.shape)
# print(input_shape)
# model.fit(X_train, Y_train, epochs=1, batch_size=10, validation_data=(X_validation, Y_validation))
