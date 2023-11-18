from google.cloud import speech
import os
import io
import os.path
import pandas as pd
import subprocess as subp
import sys

"""
TODO:
"""

def check_folder(this_dir):
    '''If {this_dir} exists, ask if okay to overwrite; otherwise, create it'''
    if not os.path.isdir(this_dir):
        os.mkdir(this_dir)


def transcribe_file(speech_file, current_lang):
    """Transcribe the given audio file."""

    if current_lang == 'English':
        lang_code = 'en-US'
    elif current_lang == 'Spanish':
        lang_code = 'es-US'
    else:
        sys.exit("Error in the language codes")

    
    client = speech.SpeechClient()
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=lang_code,
    )

    response = client.recognize(config=config, audio=audio)



    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    result_string = ''
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        result_string = str(result.alternatives[0].transcript)
        # print(u"Transcript: {}".format(result_string))

    return result_string

folder_results = '/home/luis/Dropbox/04_Audio_Perfomance_Evaluation/Google_Speech-to-Text'
folder_wav = '/home/luis/Dropbox/DATASETS_AUDIO/AOLME_bilingual/WAVS' +'/'
folder_flac = folder_wav + 'flac_audios'
transcript_path = folder_wav + 'transcript_corrected.txt'
transcript_flac_path = folder_flac + '/' + 'transcript_flac.txt'
transcript_google_path =  folder_results + '/' + 'transcript_google_results.csv'

check_folder(folder_flac)
convert_flac_flag = False

base_df = pd.read_csv(transcript_path, names=['path','GT'], sep ='\t')


if convert_flac_flag == True:
    flac_list_paths = []
    flac_list_transcript = []
    df_flac = pd.DataFrame() 
    for index, row in base_df.iterrows():
        input_path_wav = row['path']
        current_wav_name = input_path_wav.split('/')[-1]
        print(current_wav_name)
        output_path_flac = folder_flac + '/' + current_wav_name[:-4] + '.flac'
        flac_list_paths.append(output_path_flac)
        flac_list_transcript.append(row['GT'])
        cmd = f"sox {input_path_wav} --channels=1 --bits=16 {output_path_flac}"
        print(cmd)
        subp.run(cmd, shell=True)

    df_flac['path'] = flac_list_paths
    df_flac['GT'] = flac_list_transcript
    df_flac.to_csv(transcript_flac_path, mode='w', index=None, sep ='\t', header = None)

    df_flac = pd.read_csv(transcript_flac_path, names=['path','GT'], sep ='\t')


df_flac = pd.read_csv(transcript_flac_path, names=['path','GT'], sep ='\t')

google_list_paths = []
google_list_transcript = []

google_results_bilingual = []

flac_audios = sorted(os.listdir(folder_flac))
for _, row in df_flac.iterrows():
    input_flac_path = row['path']
    print(f'\n---------------------------------\n{input_flac_path}')
    current_GT = row['GT']
    current_name_noExtension = input_flac_path.split('/')[-1].split('.')[0]
    current_lang = current_name_noExtension[-11:-4]
    
    result_string = transcribe_file(input_flac_path, current_lang = current_lang)
    google_results_bilingual.append(result_string)
    print(f'GT:{current_GT} \nPr:{result_string}')

    
    google_list_paths.append(input_flac_path)
    google_list_transcript.append(current_GT)


results_df = pd.DataFrame() 
results_df['path'] = google_list_paths
results_df['GT'] = google_list_transcript

results_df['bilingual'] = google_results_bilingual
results_df.to_csv(transcript_google_path, index = False)
 