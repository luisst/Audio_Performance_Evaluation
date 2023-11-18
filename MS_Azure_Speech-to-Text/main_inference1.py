'''/**************************************************************************
    File: 01_get_original_predictions.py
    Author: Mario Esparza
    Date: 05/10/2021
    Started using Azure service on May/09/2021

    Grab audios from AOLME's test set. Run them through Microsoft's Azure
    Speech-to-text service. Save audio-name and predicted-text in a .txt file.

***************************************************************************'''
import azure.cognitiveservices.speech as speechsdk
import pandas as pd

# Txt format require path tab transcript
base_dir = '/home/luis/Dropbox/04_Audio_Perfomance_Evaluation/MS_Azure_Speech-to-Text'
inferences_file_path = base_dir + '/' + 'libriSpeechTest_azure_transcript.txt'
result_csv_path = inferences_file_path[:-4] + '_azureResults.csv'


speech_config = speechsdk.SpeechConfig(
    subscription="331aeb7dde234b2ab2522c226f4b07af",
    region="southcentralus")

base_df = pd.read_csv(inferences_file_path, names=['path','transcript'], sep ='\t')
azure_predictions = []

idx=0
wav_path ='/home/luis/Desktop/librispeech_testOthers_azure/0003.wav'

# for idx, row in base_df.iterrows():
#     wav_path = row['path']
#     auto_detect_source_language_config = \
#             speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "es-US"])

audio_input = speechsdk.AudioConfig(filename=wav_path)
print(f'\n--------------------\n{wav_path}')
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    # language='en-US',
    # auto_detect_source_language_config=auto_detect_source_language_config,
    audio_config=audio_input
)

# result = speech_recognizer.recognize_once_async().get()
speech_recognition_result = speech_recognizer.recognize_once_async().get()
# auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(speech_recognition_result)

if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("{} - Recognized: {}".format(idx, speech_recognition_result.text))

    # print("{} - {} Recognized: {}".format(idx, auto_detect_source_language_result.language, speech_recognition_result.text))
elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_recognition_result.cancellation_details
    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))
        print("Did you set the speech resource key and region values?")

azure_predictions.append(speech_recognition_result.text)
print(speech_recognition_result.text)
# print(f'{idx} - {auto_detect_source_language_result.language} : {result.text}')


base_df['azure'] = azure_predictions
base_df.to_csv(result_csv_path, index = False)

