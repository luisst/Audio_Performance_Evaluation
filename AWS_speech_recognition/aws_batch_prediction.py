from __future__ import print_function
import time
import boto3
import pandas as pd
import pathlib
import sys
sys.path.append("./../")

from my_files_utils import *

# obtain path for this current file
path_this_file = str(pathlib.Path().resolve())

name_of_job = 'testBilingual'
pred_col_name = 'bilingual'

# input folder that MIRROR bucket in S3
input_folder = r'/home/luis/Dropbox/DATASETS_AUDIO/AOLME/WAVs/Spanish/Testset_440_checked'

dataset_name = input_folder.split('/')[-1]
csv_output_path = path_this_file + '/' + dataset_name + '_aws' + '.csv'


GT_list = get_list_of_GT(input_folder)

# iterate the file names
list_of_audios = get_list_of_audios(input_folder)

aws_transcript = []
aws_paths = []

num_total = len(list_of_audios)
cnt = 0

for current_path in list_of_audios:
    current_name = current_path.split('/')[-1]
    print(current_name)

    job_name = "{}_440_aolme_{}".format(name_of_job, cnt)
    job_uri = "s3://myaolmebucket/{}".format(current_name)
    transcribe = boto3.client('transcribe')
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        # LanguageCode='es-ES',
        IdentifyLanguage=True,
        LanguageOptions=['en-US'|'es-ES'|'es-US'],
        # JobExecutionSettings = {"AllowDeferredExecution": True},
        Settings = {"ShowAlternatives": True, "MaxAlternatives": 4}
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == "COMPLETED":
        data = pd.read_json(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        print("{} / {} completed".format(cnt, num_total))
        my_transcription = data['results'][2][0]['transcript']
    elif status['TranscriptionJob']['TranscriptionJobStatus'] == "FAILED":
        print("Transcript Failed")
        my_transcription = ""

    cnt = cnt + 1

    aws_transcript.append(my_transcription)
    aws_paths.append(current_path)


columns = ['path', 'GT', pred_col_name]
write_my_csv(aws_paths, GT_list, aws_transcript, path=csv_output_path, cols = columns)
