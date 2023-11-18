
import pandas as pd
from pathlib import Path

list_start_time = []
list_end_time = []
list_speaker = []
list_lang = []
list_transcription = []
list_confidence = []

path_this_file = Path().resolve().joinpath('scratch_json_file.json')
data = pd.read_json(path_this_file)
number_speakers = data['results']['speaker_labels']['speakers']
all_items = data['results']['items']

for current_entry in all_items:
    if current_entry['type'] == 'pronunciation':
        list_start_time.append(current_entry['start_time'])
        list_end_time.append(current_entry['end_time'])
        list_speaker.append(current_entry['speaker_label'])
        list_lang.append(current_entry['language_code'])

        list_transcription.append(current_entry['alternatives'][0]['content'])
        list_confidence.append(current_entry['alternatives'][0]['confidence'])