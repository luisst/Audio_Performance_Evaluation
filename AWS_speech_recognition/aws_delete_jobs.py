from __future__ import print_function
import time
import boto3

name_of_job = 'test'
last_number_job = 439
cnt = 0

for cnt in range(0, last_number_job + 1):
    job_name = "{}_440_aolme_{}".format(name_of_job, cnt)
    transcribe = boto3.client('transcribe')

    transcribe.delete_transcription_job(
        TranscriptionJobName=job_name,
    )
