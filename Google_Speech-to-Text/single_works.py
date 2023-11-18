# Imports the Google Cloud client library
from google.cloud import speech
import io


speech_file_path = 'minitest.wav'

# Instantiates a client
client = speech.SpeechClient()

with io.open(speech_file_path, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

first_lang = "en-US"
second_lang = "es-US"

# first_lang = "es-US"
# second_lang = "en-US"


config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code=first_lang,
    alternative_language_codes=[second_lang],
)

# Detects speech in the audio file
response = client.recognize(config=config, audio=audio)

for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript)) 
