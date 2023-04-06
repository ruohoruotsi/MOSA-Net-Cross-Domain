import numpy as np
import librosa
import scipy.io
import scipy.stats


# Please follow the following format to make the input list
# PESQ score, STOI score, SDI score, filepath directory
def input_list_read(filelist):
    f = open(filelist, 'r')
    Path = []
    for line in f:
        Path = Path + [line[0:-1]]
    return Path


def spectrogram_and_phase(path):
    audio_data, _ = librosa.load(path, sr=16000, mono=True)
    if np.max(abs(audio_data)) != 0:
        audio_data = audio_data / np.max(abs(audio_data))

    F = librosa.stft(audio_data, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    Lp = np.abs(F)  # magnitude spectrogram
    NLp = Lp
    NLp = np.reshape(NLp.T, (1, NLp.shape[1], 257))
    end2end = np.reshape(audio_data, (1, audio_data.shape[0], 1))
    return NLp, end2end


def norm_data(input_x):
    input_x = (input_x - 0) / (5 - 0)
    return input_x


def denorm(input_x):
    input_x = input_x * (5 - 0) + 0
    return input_x
