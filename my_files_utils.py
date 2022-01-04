#!/usr/bin/env python3
"""
Utilities functions to check and find files from UNIX-based systems

This should be syncronized for all of my repos

"""

import glob
import os
import sys
import shutil
import re
import numpy as np
import pandas as pd
from os.path import join as pj
from os.path import exists
import collections
import datetime

## List of the functions:

# - check_empty_folder .-

###   #   #   #       Check processes        #  #  #

def check_empty_folder(current_folder_path):
    """ Return true if folder is NOT empty """
    if (len(os.listdir(current_folder_path)) != 0):
       return True
    else:
        print("Folder {} is empty".format(current_folder_path))
        return False


def check_ending_format(current_file_path, ending_format_substring):
    if current_file_path[-3:] != ending_format_substring:
        print("Error in the format, must end with {}!".format(ending_format_substring))
        sys.exit()


def check_same_length(list1, list2):
    if len(list1) != len(list2):
        print("Error, your list1 and list2 have different lengths")
        sys.exit()


def check_csv_exists(csv_path):
    if exists(csv_path):
        print("CSV file already exists, do you want to overwrite? (y)")
        if input().lower() != 'y':
            print("File not modified")
            sys.exit()



###   #   #   #         Store TXT and CSV          #  #  #


def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)

    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')


def write_my_csv(*args, **kwargs):
    """
    Function to write csv files.
    args:
        - Columns for the csv (matched to the names)
    kwargs:
        - cols: List of names for columns (matched to args)
        - path: output_path for the csv
    """


    # my_df = pd.DataFrame(index=False)
    my_df = pd.DataFrame()

    csv_path = kwargs['path']
    columns_values = kwargs['cols']

    # check if csv file exists
    check_csv_exists(csv_path)

    if len(args) > 2:
        check_same_length(args[0], args[1])
    elif len(args) > 3:
        check_same_length(args[1], args[2])

    idx = 0
    for current_list in args:
        my_df[columns_values[idx]] = current_list
        idx = idx + 1

    today_date = '_' + str(datetime.date.today())
    full_output_csv_path = '/'.join(csv_path.split('/')[0:-1]) + '/' + csv_path.split('/')[-1][0:-4] + today_date + '.csv'
    my_df.to_csv(csv_path)


###   #   #   #       Files processes        #  #  #



def locate_single_txt(src_dir, obj_format = 'txt'):
    """
    Input:
    - Folder path

    Output:
    - Path of the only transcript found in folder
    """

    all_texts = glob.glob("{}/*.{}".format(src_dir, obj_format))

    if len(all_texts) != 1:
        print("Too many transcripts! Please use only 1")
    else:
        input_transcript_path = all_texts[0]
        print(input_transcript_path)
        print("\n")

    return input_transcript_path


def get_pathsList_from_transcript(transcript_path, csv_flag=False):
    """
        Given a transcript in txt (or csv), it returns a list of the paths of all the audios
    """

    if csv_flag == False:
        check_ending_format(transcript_path, 'txt')

        transcript_data = pd.read_csv(transcript_path, sep="\t", header=None, index_col=False)
        path_list = transcript_data[0].tolist()
        return path_list


def get_list_of_GT(folder_path, csv_flag=False):

    transcript_path = locate_single_txt(folder_path)

    if csv_flag == False:
        check_ending_format(transcript_path, 'txt')

        transcript_data = pd.read_csv(transcript_path, sep="\t", header=None, index_col=False)
        GT_list = transcript_data[1].tolist()
        return GT_list


def get_list_of_audios(folder_path, audio_extension = 'wav',
                       confirm_with_transcript = True,
                       verbose = False):
    """
    Confirm_with_transcript requires: file path in first column
    To-do: check if in csv file the header_none will throw count the header as the first row.
    To-do: Add a function to only compare the filename, not the entire path
    """
    all_audios = sorted(glob.glob("{}/*.{}".format(folder_path, audio_extension)))

    ### To double check with transcript paths
    if confirm_with_transcript:
        transcript_path = locate_single_txt(folder_path)

        # Obtain list of paths from transcript
        transcript_lists = sorted(get_pathsList_from_transcript(transcript_path))

        # Compare 2 lists, full path or only names
        if collections.Counter(all_audios) == collections.Counter(transcript_lists):
            if verbose:
                print("Transcript and audios are consistent")
        else:
            print("Transcript paths and names of files does not match")
            sys.exit()

    return all_audios
