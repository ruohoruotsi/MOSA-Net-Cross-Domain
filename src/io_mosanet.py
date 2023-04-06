"""
Author: iroro
"""

import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import keras
from keras.models import Model
from keras.layers import concatenate
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling1D, AveragePooling1D
from keras.layers import TimeDistributed, Bidirectional, Input, LSTM
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import numpy as np
import random
import utils

random.seed(999)


def BLSTM_CNN_with_ATT_VoiceMOS():
    input_size = (None, 1)
    _input = Input(shape=(None, 257))
    _input_end2end = Input(shape=(None, 1))

    SincNet_ = Sinc_Conv_Layer(input_size, N_filt=257, Filt_dim=251, fs=16000, NAME="SincNet_2").compute_output(
        _input_end2end)
    merge_input = concatenate([_input, SincNet_], axis=1)
    re_input = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input)

    # CNN
    conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv1 = (Conv2D(16, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv1)

    conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv2)

    conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv3)

    conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv4)

    re_shape = keras.layers.core.Reshape((-1, 4 * 128), input_shape=(-1, 4, 128))(conv4)
    _input_hubert = Input(shape=(None, 1024))
    mean_polling = AveragePooling1D(pool_size=2, strides=1, padding='same')(_input_hubert)
    bottleneck = TimeDistributed(Dense(512))(mean_polling)
    concat_with_wave2vec = concatenate([re_shape, bottleneck], axis=1)
    blstm = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1 = Dropout(0.3)(dense1)

    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                 kernel_regularizer=keras.regularizers.l2(1e-4),
                                 bias_regularizer=keras.regularizers.l1(1e-4), attention_regularizer_weight=1e-4,
                                 name='Attention')(dense1)
    Frame_score = TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score')(attention)
    MOS_score = GlobalAveragePooling1D(name='MOS_score')(Frame_score)

    model = Model(outputs=[MOS_score, Frame_score], inputs=[_input, _input_end2end, _input_hubert])

    return model


def voicemos_inference(utterance_list, utterance_hubert_feat_list):
    print("\n[Dialog Intel] loading the VoiceMOS MOSA-Net model ...")

    model_test = BLSTM_CNN_with_ATT_VoiceMOS()
    model_test.load_weights('../pretrained_models/MOSA-Net_Cross_Domain_epoch_100.h5')

    print("\n[Dialog Intel] evaluating utterance list ...")
    mos_predict = np.zeros([len(utterance_list), ])
    data_path = '../data/wav/'

    utterances_list = list(set([os.path.splitext(item)[0] for item in utterance_list]))
    mos_predict_utterances = {system: [] for system in utterances_list}

    for i in range(len(utterance_list)):
        asessment_filepath = utterance_list[i].split(',')
        hubert_filepath = utterance_hubert_feat_list[i].split(',')
        wav_name = asessment_filepath[0]

        complete_path = data_path + wav_name
        noisy_spectrogram, noisy_waveform = utils.spectrogram_and_phase(complete_path)
        noisy_hubert = np.load(hubert_filepath[1])  # hubert load extracted features

        [voice_mos_score, frame_mos] = model_test.predict([noisy_spectrogram, noisy_waveform, noisy_hubert],
                                                          verbose=0, batch_size=1)

        denorm_mos_predict = utils.denorm(voice_mos_score)
        mos_predict[i] = denorm_mos_predict
        system_names = os.path.splitext(wav_name)[0]
        mos_predict_utterances[system_names].append(denorm_mos_predict[0])

        estimated_score = denorm_mos_predict[0]
        info = asessment_filepath[0] + ', VoiceMOS: ' + str(estimated_score[0])
        print(
            "[{}] info: {}   spectrogram shape: {}  waveform shape: {}  hubert feature shape: {}   frame_mos shape: {}".
            format(i, info, noisy_spectrogram.shape, noisy_waveform.shape, noisy_hubert.shape, frame_mos.shape))


# hubert model: hubert_large_ll60k.pt downloaded from
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert


if __name__ == '__main__':
    voicemos_inference(utils.input_list_read('../data/io_test_list.txt'),
                       utils.input_list_read('../data/List_Npy_Val_hubert_MOS_Challenge_phase1_main.txt'))
