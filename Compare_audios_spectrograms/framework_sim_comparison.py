import numpy as np
import sounddevice as sd
import random
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from librosa.display import specshow
from librosa.feature import melspectrogram as MelSpec

from DA_pyroom3 import DAwithPyroom

fs=16000


def folderWavs_to_npy(output_wav_directory, output_npy_directory):
    X_data = []
    i = 0
    for item in sorted(os.listdir(output_wav_directory)):
        if item[-4:] == ".wav":
            wav_path = output_wav_directory + r'/' + item
            raw_data, samplerate = sf.read(wav_path)
            print("{} item {}, shape {}".format(i, item, str(raw_data.shape)))

            X_data.append(raw_data)

            if (np.count_nonzero(raw_data) == 0):
                print("All zeros. {} item {}".format(i, item))
            i = i + 1

    print("Length of X_data noises is {}".format(len(X_data)))

    # SAVE GT
    np.save(output_npy_directory, X_data)


def listen_audio(idx, x_data):
    my_audio = x_data[idx]
    sd.play(my_audio, fs)


def check_same_format(audio1, audio2):
    type_audio1 = audio1.dtype.kind

    if audio2.dtype.kind != type_audio1:
        raise TypeError("'dtype' must be a floating point type")
    print("Both audios are {}".format(type_audio1))


def listen_2_audios(audio1, audio2, repeat = True):

    audio1 = np.trim_zeros(audio1)
    audio2 = np.trim_zeros(audio2)

    # Check both audio can be concatenated
    check_same_format(audio1, audio2)

    # Obtain biggest audio and create silence
    longest_audio = np.maximum(len(audio1), len(audio2))
    silence_part = np.zeros((longest_audio,), dtype=float)

    # Concatenate audios according to repeat
    if repeat:
       final_audio = np.concatenate((audio1, silence_part, audio2, silence_part, audio1))
    else:
       final_audio = np.concatenate((audio1, silence_part, audio2))

    sd.play(final_audio, fs)


def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range"""
    max_val, min_val = matrix.max(), matrix.min()
    normalized = np.divide(np.subtract(matrix, min_val), (max_val - min_val))
    return normalized


wav_directory = '/home/luis/Dropbox/dataset_AOLME_pipelineLuis/01c_AOLME_dataset_checked'
output_npy_directory = '/home/luis/Dropbox/dataset_TTS_pipelineLuis/05_Pyroom_NPY/AOLME440_testset.npy'

read_wavs_flag = False
if read_wavs_flag:
    folderWavs_to_npy(wav_directory, output_npy_directory)

# 1) Load the npys: clean and the dirty ones
input_path_sim_1 = '/home/luis/Dropbox/dataset_TTS_fromTxtTestset/05_Pyroom_NPY/Npy_input/DataAugmented/TS1_lupe_spanish_DA_0_bs_tab.npy'
input_path_sim_2 = '/home/luis/Dropbox/dataset_TTS_fromTxtTestset/05_Pyroom_NPY/Npy_input/DataAugmented/TS1_lupe_spanish_DA_1_bs_tab.npy'
input_path_sim_3 = '/home/luis/Dropbox/dataset_TTS_fromTxtTestset/05_Pyroom_NPY/Npy_input/DataAugmented/TS1_lupe_spanish_DA_2_bs_tab.npy'

input_path_raw = '/home/luis/Dropbox/dataset_TTS_fromTxtTestset/05_Pyroom_NPY/Npy_input/TS1_lupe_spanish.npy'
input_GT_aolme = output_npy_directory

my_dataset_simulated_1 = np.load(input_path_sim_1, allow_pickle=True)
my_dataset_simulated_2 = np.load(input_path_sim_2, allow_pickle=True)
my_dataset_simulated_3 = np.load(input_path_sim_3, allow_pickle=True)

my_dataset_clean = np.load(input_path_raw, allow_pickle=True)
my_dataset_GT = np.load(input_GT_aolme, allow_pickle=True)

# Here I can separate them into S0-S3 (position) and TTS voices (voices).
indx_audio = 58
# indx_audio = random.randint(0, my_dataset_clean.shape[0]-1)

my_audio_clean_long = my_dataset_clean[indx_audio]
my_audio_clean = np.trim_zeros(my_audio_clean_long)

audio_GT = my_dataset_GT[indx_audio]
audio_sim1 = my_dataset_simulated_1[indx_audio]

audio_sim1_fast1 = librosa.effects.time_stretch(audio_sim1, 1.1)

# Manually listen to both audios one after the other
# listen_2_audios(indx_audio, my_dataset_clean, my_dataset_simulated_1, repeat = True)

listen_2_audios(audio_GT, audio_sim1_fast1, repeat = True)

sr = 16000
n_mels = 128
n_fft = 448
hop_length = n_fft //2

x_axis_time1 = np.linspace(0, len(audio_GT) / fs, num = len(audio_GT))
x_axis_time2 = np.linspace(0, len(audio_sim1_fast1) / fs, num = len(audio_sim1_fast1))


spec1 = MelSpec(y=audio_GT, sr=sr, n_mels=n_mels, n_fft=n_fft,\
                            hop_length=hop_length)
spec1 = normalize_0_to_1(spec1)
S_dB1 = librosa.power_to_db(spec1, ref=np.max)

spec2 = MelSpec(y=audio_sim1_fast1, sr=sr, n_mels=n_mels, n_fft=n_fft,\
                            hop_length=hop_length)
spec2 = normalize_0_to_1(spec2)
S_dB2 = librosa.power_to_db(spec2, ref=np.max)

fig, ax = plt.subplots(2, 2, figsize=(20, 11))
# fig.tight_layout(pad=3qq)
ax[0,0].set_xlabel('Time', fontsize=14)
ax[0,0].tick_params(axis='both', which='major', labelsize=10)
ax[0,0].set_title("audio 1", fontsize=17)
ax[0,0].plot(x_axis_time1, audio_GT, 'r')


ax[1,0].set_xlabel('Time', fontsize=14)
ax[1,0].set_title("audio 2", fontsize=17)
ax[1,0].plot(x_axis_time2, audio_sim1_fast1, 'g')

ax[0,1].set_xlabel('Specs', fontsize=14)
ax[0,1].set_title("audio 1", fontsize=17)
img = specshow(S_dB1, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[0,1])
fig.colorbar(img, ax=ax[0,1], format='%+2.0f dB')

ax[1,1].set_xlabel('Specs', fontsize=14)
ax[1,1].set_title("audio 2", fontsize=17)
img = specshow(S_dB2, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1,1])
fig.colorbar(img, ax=ax[1,1], format='%+2.0f dB')

plt.show()

# share y axis on audios
# set same x axis on both audios
# increase font of numbers
# set the title of 



def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


namestr(audio_GT, globals())

# # plot using code below the spectrogram + audio plot. clean vs dirtyA / dirtyB

# def check_audios(self, longest, HP):
#     """Make sure that all audios' spectrograms return the same shape"""
#     #We only have to worry of the "time" dimension. "Frequency" will
#     #always return the value in HP['mels'].
#     Dims = np.zeros(len(self.audios_paths), dtype=int)
#     for idx, audio_path in enumerate(self.audios_paths):
#         y, _ = librosa.load(audio_path, sr=HP['sr'])
#         #If size of audio is smaller than longest, do some padding
#         if y.size < longest:
#             y = pad_audio(audio_path, HP['sr'], longest)

#         #Get spectrogram and dimensions
#         spec = MelSpec(y=y, sr=HP['sr'], n_mels=HP['mels'], n_fft=HP['N'],
#                         hop_length=HP['HL'])

#         Dims[idx] = spec.shape[1]

#     indices = (Dims != Dims[0]).nonzero()
#     if not indices[0].size:
#         print("You are good to go, all audios' spectrograms have the same"
#                 " shape (dimensions).")
#     else:
#         print("ERROR: one or more spectrograms returned a different shape"
#                 ". Here is the list of audio(s) that caused this issue:")

#         for i in range(0, indices[0].size):
#             print("f{indices1[0][i]}, ", end="")
#         print("\nBye.")

#     return spec.shape

# def pad_audio(audio_path, SR, longest):
#     """Pad audio with zeros to match length of {longest}"""
#     y, sr = librosa.load(audio_path, SR)
#     placeholder = np.zeros(longest)
#     placeholder[:y.size] = y
#     return placeholder

# def normalize_0_to_1(matrix):
#     """Normalize matrix to a 0-to-1 range"""
#     max_val, min_val = matrix.max(), matrix.min()
#     normalized = np.divide(np.subtract(matrix, min_val), (max_val - min_val))
#     return normalized

# # # Initialize X
# # X = np.zeros((self.bs, self.dim1, self.dim2), dtype=np.float32)

# # y = pad_audio(audio_path, self.sr, self.longest)

# #Hyper Parameters
# HP = {
#       'mels': 128,
#       'epochs': 3,
#       'sr': 16000,
#       'inChannel' : 1, #required for first CNN layer
#       'lr': 3e-4, #learning rate
#       'es': {'monitor': 'val_loss', 'patience': 10, 'mode': 'min'}, #early stop
#       #N and HL values are used to mimic torchaudio.transforms.MelSpectrogram
#       'N': 448 #length of the FFT window, originally, it was 400
# }
# HP['HL'] = HP['N'] // 2 #hop length

# # y, _ = librosa.load(audio_path, sr=HP['sr'])

# spec = MelSpec(y=my_audio_clean, sr=HP['sr'], n_mels=HP['mels'], n_fft=HP['N'],
#                            hop_length=HP['HL'])

# # spec = normalize_0_to_1(spec)
# # spec = normalize_0_to_1(spec)


# fig, ax = plt.subplots()
# S_dB = librosa.power_to_db(spec, ref=np.max)
# img = specshow(S_dB, x_axis='time', y_axis='mel', sr=HP['sr'], fmax=8000, ax=ax)

# #
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# # ax.set(title=title)
# # plt.savefig(fig_name)
# plt.show()

# save it as wav file to share it properly.

# I need a proper cfg file to test the gain levels and inyection of background noise. can i do it offline?

# DA2 allows me to run my code with only 1 audio. that can be another usage of this framework. simulation on the fly.



