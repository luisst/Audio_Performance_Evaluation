#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:19:57 2022

@author: luis
"""

from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# wav_path='/home/luis/Dropbox/DATASETS_AUDIO/AOLME_bilingual/WAVS/G-C2L1P-Apr12-A-Allan_q2_02-05_Spanish_001.wav'
wav_path=Path.cwd().joinpath('G-C2L1W-Feb27-B-Issac_q2_03-04_RNDshort-014.wav')

speech_config = speechsdk.SpeechConfig(subscription='769ed07f4cc845c6b49bd5db3e3dc05a', region='southcentralus' )
    
    
audio_input = speechsdk.AudioConfig(filename='miniaudio.wav')

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

result = speech_recognizer.recognize_once_async().get()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized: {}".format(result.text))
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized: {}".format(result.no_match_details))
elif result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = result.cancellation_details
    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))
        print("Did you set the speech resource key and region values?")