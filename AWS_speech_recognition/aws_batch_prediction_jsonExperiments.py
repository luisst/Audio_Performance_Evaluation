from __future__ import print_function
import time
import boto3
import pandas as pd
import sys
sys.path.append("./../")
from pathlib import Path

from my_files_utils import *


# obtain path for this current file
path_this_file = Path().resolve()
name_bucket = 'sdtest23'
name_of_job = 'testset'
pred_col_name = 'multiLang'

# input folder that MIRROR bucket in S3
input_wav_pth = Path.home().joinpath('Dropbox','DATASETS_AUDIO','AOLME_SD_Collection','TestSet','00_Single_videos','wav_output')
output_parent_pth = Path.home().joinpath('Dropbox','DATASETS_AUDIO','AOLME_SD_Collection','TestSet','Commercial_results','aws')
dataset_name = input_wav_pth.stem + '_aws' + '.csv'
csv_output_path = output_parent_pth.joinpath(dataset_name)


# GT_list = get_list_of_GT(input_wav_pth)

# iterate the file names
list_of_audios = sorted(list(input_wav_pth.glob('*.wav')))

aws_paths = []
list_start_time = []
list_end_time = []
list_speaker = []
list_lang = []
list_transcription = []
list_confidence = []

num_total = len(list_of_audios)
cnt = 0

for current_path in list_of_audios:
    current_name = current_path.name
    print(current_name)

    job_name = f'{name_of_job}_sdtest23_{cnt}'
    job_uri = f's3://{name_bucket}/{current_name}'
    transcribe = boto3.client('transcribe')
    
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        # LanguageCode='es-ES',
        IdentifyMultipleLanguages=True,
        # IdentifyLanguage=True,
        LanguageOptions=['en-US', 'es-ES', 'es-US'],
        # JobExecutionSettings = {"AllowDeferredExecution": True},
        Settings = {"ShowAlternatives": True, "MaxAlternatives": 4, "ShowSpeakerLabels": True, "MaxSpeakerLabels": 5, "ChannelIdentification": False}
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == "COMPLETED":
        data = pd.read_json(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])

        number_speakers = data['results']['speaker_labels']['speakers']
        all_items = data['results']['items']

        print(f'{cnt} / {num_total} | Number of speakers found: {number_speakers}')

        for current_entry in all_items:
            if current_entry['type'] == 'pronunciation':
                list_start_time.append(current_entry['start_time'])
                list_end_time.append(current_entry['end_time'])
                list_speaker.append(current_entry['speaker_label'])
                list_lang.append(current_entry['language_code'])
                list_transcription.append(current_entry['alternatives'][0]['content'])
                list_confidence.append(current_entry['alternatives'][0]['confidence'])
                aws_paths.append(current_name)

    elif status['TranscriptionJob']['TranscriptionJobStatus'] == "FAILED":
        print("Transcript Failed")
        my_transcription = ""


    cnt = cnt + 1

                

columns = ['path', 'speaker', 'lang_code', 'start_time', 'end_time', 'prediction', 'confidence']

write_2_csv(aws_paths, list_speaker, list_lang, list_start_time, list_end_time, list_transcription ,list_confidence, path=csv_output_path, cols = columns, txt_flag =  True)

