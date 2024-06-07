import requests

URL = "http://127.0.0.1:5050/predict"
TEST_AUDIO_FILE = "test/bed.wav"

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE,"rb")
    values = {"file":(TEST_AUDIO_FILE,audio_file,"audio/wav")}
    response = requests.post(URL,files =values)
    data = response.json()
    print(f"Predicted word is: {data['keyword']}")
