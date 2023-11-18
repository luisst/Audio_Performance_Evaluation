import sys
from pathlib import Path


def change_aws_spkr(speaker_ID):
    if speaker_ID == 'spk_0':
        return 'S0'
    elif speaker_ID == 'spk_1':
        return 'S1'
    elif speaker_ID == 'spk_2':
        return 'S2'
    elif speaker_ID == 'spk_3':
        return 'S3'
    elif speaker_ID == 'spk_4':
        return 'S4'
    else:
        sys.exit("Speaker not recognized")

def change_aws_lang(speaker_lang):
    if speaker_lang == 'en-US':
        return 'Eng'
    elif speaker_lang == 'es-US':
        return 'Spa'
    elif speaker_lang == 'es-ES':
        return 'Spa'
    else:
        sys.exit("Lang code not recognized")

def read_lines_webapp(line):
    speaker_IDLang, strt_time, end_time = line.split('\t')
    speaker_ID = speaker_IDLang[:2]
    speaker_lang = speaker_IDLang[2:]
    end_time = end_time.strip()
    return speaker_ID, speaker_lang, strt_time, end_time


def read_lines_aws(line):
    name_audio, speaker_ID, speaker_lang, strt_time, end_time, gt_txt, confidence_val = line.split('\t')
    end_time = end_time.strip()
    speaker_ID = change_aws_spkr(speaker_ID)
    speaker_lang = change_aws_lang(speaker_lang)
    return speaker_ID, speaker_lang, strt_time, end_time


include_header = False
csv_flag = False
case_process = 'aws' # from aws, webapp, 

# Read all txt (from AWS) | csv from Webapp files from folder_pth
folder_pth = Path.home().joinpath('Dropbox', '04_Audio_Perfomance_Evaluation','AWS_speech_recognition','results_aws_Dec22')

if csv_flag:
    my_suffix = 'csv'
else:
    my_suffix = 'txt'

for csv_pth in folder_pth.glob(f'*.{my_suffix}'):
    print( csv_pth )
    # Src folder with all csv file to transform (future)
    csv_name = csv_pth.stem

    # Load 1 csv file
    f = open(csv_pth, 'r')
    lines = f.readlines()
    f.close()

    new_file = open(csv_pth, "w")

    gt_dict = {}

    if include_header:
        lines.pop(0)

    for line in lines:

        if case_process == 'aws':
            speaker_ID, speaker_lang, strt_time, end_time = read_lines_aws(line)
        elif case_process == 'webapp':
            speaker_ID, speaker_lang, strt_time, end_time = read_lines_webapp(line)
        
        speaker_lang_ID = speaker_ID + speaker_lang


        if speaker_lang_ID not in gt_dict:
            gt_dict[speaker_lang_ID] = [[speaker_ID, speaker_lang, strt_time, end_time]]
        else:
            gt_dict[speaker_lang_ID].append([speaker_ID, speaker_lang, strt_time, end_time])

    for key in gt_dict.keys():
        current_speaker_intervals = gt_dict[key]

        # Sort using the start_time
        current_speaker_intervals = sorted(current_speaker_intervals, key=lambda x: float(x[2]))
        stack = []
        # insert first interval into stack
        stack.append(current_speaker_intervals[0])

        for i in current_speaker_intervals[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if stack[-1][2] <= i[2] <= stack[-1][3]:
                stack[-1][3] = max(stack[-1][3], i[3])
            else:
                stack.append(i)
    
        print("\n\nThe Merged Intervals are :", end=" ")
        for i in range(len(stack)):
            print(stack[i], end=" ")


        # Write the ordered speaker GT
        for Sx in stack:
            new_file.write(f'{Sx[0]}{Sx[1]}\t{Sx[2]}\t{Sx[3]}\n')
        
    new_file.close()
