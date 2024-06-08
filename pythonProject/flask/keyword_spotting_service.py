import tensorflow.keras as keras
import numpy as np
import librosa

SAMPLE_NUM = 22050
class _Keyword_Spotting_Service():
    model = None
    instance = None
    mappings = [
        "backward",
        "bed",
        "bird",
        "cat"
    ]

    def predict(self,file_path):

        # Extract MFCCs from the audio file (# samples,# segments, #n_mfccs)
        MFCCs = self.preprocess(file_path)

        # Convert the 2d array MFCCs to 4d array (# samples, # segments, # n_mfccs, # n_channels)
        MFCCs = MFCCs[np.newaxis,...,np.newaxis]

        # Do predictions
        num_predictions = self.model.predict(MFCCs) # (# samples, # num_labels)
        prediction = np.argmax(num_predictions)
        predicted_keyword = self.mappings[prediction]
        return predicted_keyword

    def preprocess(self,file_path,n_mfcc=13,n_fft=2048,hop_length=512):
        # load file
        signal,sr = librosa.load(file_path,sr = 22050)

        # Check for consistency
        if len(signal) > SAMPLE_NUM:
            signal = signal[:SAMPLE_NUM]

        # Extract mfccs
        MFCC = librosa.feature.mfcc(y=signal,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
        return MFCC.T

def Keyword_Spotting_Service():
    if _Keyword_Spotting_Service.instance == None:
        _Keyword_Spotting_Service.instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model("model.keras")
    return _Keyword_Spotting_Service.instance

if __name__ == "__main__":
    keyword_service = Keyword_Spotting_Service()
    predicted_word = keyword_service.predict("test/bed.wav")
    print(predicted_word)