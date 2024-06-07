import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensor
DATASET_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

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
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,
                                                                 test_size=validation_size)

    # convert inputs from 2d to 3d for CNN
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    return X_train,X_validation,X_test,Y_train,Y_validation,Y_test

def build_model(input_shape,learning_rate):

    # Build Network

def main():
    # Load train data splits
    # X_train,X_validation,X_test,Y_train,Y_validation,Y_test = get_data_splits(DATASET_PATH)

    # Build CNN model
    # input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]) # (n_fft * n_mfccs * channels)
    # model = build_model(input_shape,learning_rate = LEARNING_RATE)

    # Train the model
    # model.fit(X_train,Y_train,epochs=EPOCHS,batch_size = BATCH_SIZE, validation_data = (X_validation,Y_validation)

    # Evaluate/test the model
    # test_error,test_accuracy = model.evaluate(X_test,Y_test)
    # print(f"test_error: {test_error}\n test_accuracy: {test_accuracy}")

    # Save the model
    # model.save(SAVED_MODEL_PATH)