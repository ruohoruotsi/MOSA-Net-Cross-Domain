"""
@author: Ryandhimas Zezario, iroro
ryandhimas@citi.sinica.edu.tw
"""

import os, sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
import math

# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling1D, AveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, LSTM
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time
import numpy as np
import numpy.matlib
import random
import utils

random.seed(999)


def BLSTM_CNN_with_ATT_cross_domain():
    input_size = (None, 1)
    _input = Input(shape=(None, 257))
    _input_end2end = Input(shape=(None, 1))

    SincNet_ = Sinc_Conv_Layer(input_size, N_filt=257, Filt_dim=251, fs=16000, NAME="SincNet_1").compute_output(
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
    bottleneck = TimeDistributed(Dense(512, activation='relu'))(_input_hubert)
    concat_with_wave2vec = concatenate([re_shape, bottleneck], axis=1)
    blstm = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1 = Dropout(0.3)(dense1)

    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                 kernel_regularizer=keras.regularizers.l2(1e-4),
                                 bias_regularizer=keras.regularizers.l1(1e-4), attention_regularizer_weight=1e-4,
                                 name='Attention')(dense1)
    Frame_score = TimeDistributed(Dense(1), name='Frame_score')(attention)
    PESQ_score = GlobalAveragePooling1D(name='PESQ_score')(Frame_score)

    attention2 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                  kernel_regularizer=keras.regularizers.l2(1e-4),
                                  bias_regularizer=keras.regularizers.l1(1e-4), attention_regularizer_weight=1e-4,
                                  name='Attention2')(dense1)
    Frame_stoi = TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_stoi')(attention2)
    STOI_score = GlobalAveragePooling1D(name='STOI_score')(Frame_stoi)

    attention3 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                  kernel_regularizer=keras.regularizers.l2(1e-4),
                                  bias_regularizer=keras.regularizers.l1(1e-4), attention_regularizer_weight=1e-4,
                                  name='Attention3')(dense1)
    Frame_sdi = TimeDistributed(Dense(1), name='Frame_sdi')(attention3)
    SDI_score = GlobalAveragePooling1D(name='SDI_score')(Frame_sdi)

    model = Model(outputs=[PESQ_score, Frame_score, STOI_score, Frame_stoi, SDI_score, Frame_sdi],
                  inputs=[_input, _input_end2end, _input_hubert])

    return model


def cross_domain_inference(utterance_list, utterance_hubert_feat_list):
    print("\n[Dialog Intel] loading the {PESQ, SDI, STOI} MOSA-Net model ...")
    data_path = '../data/wav/'

    model = BLSTM_CNN_with_ATT_cross_domain()
    model.load_weights('../pretrained_models/MOSA_Net_Cross_Domain_Multi_Target.h5')

    print("\n[Dialog Intel] evaluating utterance list ...")
    pesq_predict = np.zeros([len(utterance_list), ])
    stoi_predict = np.zeros([len(utterance_list), ])
    sdi_predict = np.zeros([len(utterance_list), ])

    pesq_predict_frame = []
    stoi_predict_frame = []
    sdi_predict_frame = []

    for i in range(len(utterance_list)):
        utt_filepath = utterance_list[i].split(',')
        hubert_filepath = utterance_hubert_feat_list[i].split(',')
        noisy_spectrogram, noisy_waveform = utils.spectrogram_and_phase(data_path + utt_filepath[0])
        noisy_hubert = np.load(hubert_filepath[1])

        [PESQ_score, frame_score, STOI_score, frame_stoi, SDI_score, frame_sdi] = model.predict(
            [noisy_spectrogram, noisy_waveform, noisy_hubert], verbose=0, batch_size=1)
        pesq_predict[i] = PESQ_score
        stoi_predict[i] = STOI_score
        sdi_predict[i] = SDI_score
        print("[{}] {}   PESQ: {}    SDI: {}     STOI: {}".format(i, utt_filepath, PESQ_score, SDI_score, STOI_score))

        pesq_predict_frame.extend(frame_score.flatten())
        stoi_predict_frame.extend(frame_stoi.flatten())
        sdi_predict_frame.extend(frame_sdi.flatten())


    # Are results linear or monotonic?
    print("-" * 65)
    #################################################################################
    # Plotting the scatter plot AVERAGE PESQ vs STOI
    LCC = np.corrcoef(pesq_predict, stoi_predict)
    print("Average PESQ vs Average STOI: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(pesq_predict.T, stoi_predict.T)
    print("Average PESQ vs Average STOI: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(pesq_predict, stoi_predict, s=14)
    plt.xlabel('Average Predicted PESQ')
    plt.ylabel('Average Predicted STOI')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Avg_PESQ_vs_STOI_MOSA_Net_Cross_Domain.png', dpi=150)

    # Plotting the scatter plot FRAME PESQ vs STOI
    LCC = np.corrcoef(pesq_predict_frame, stoi_predict_frame)
    print("Frame PESQ vs Frame STOI: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(np.array(pesq_predict_frame).T, np.array(stoi_predict_frame).T)
    print("Frame PESQ vs Frame STOI: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(pesq_predict_frame, stoi_predict_frame, s=14)
    plt.xlabel('Predicted PESQ Frame scores')
    plt.ylabel('Predicted STOI Frame scores')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Frame_PESQ_vs_STOI_MOSA_Net_Cross_Domain.png', dpi=150)

    print("-" * 65)
    #################################################################################
    # Plotting the scatter plot AVERAGE SDI vs STOI
    LCC = np.corrcoef(sdi_predict, stoi_predict)
    print("Average SDI vs Average STOI: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(sdi_predict.T, stoi_predict.T)
    print("Average SDI vs Average STOI: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(sdi_predict, stoi_predict, s=14)
    plt.xlabel('Average Predicted SDI')
    plt.ylabel('Average Predicted STOI')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Avg_SDI_vs_STOI_MOSA_Net_Cross_Domain.png', dpi=150)

    # Plotting the scatter plot FRAME SDI vs STOI
    LCC = np.corrcoef(sdi_predict_frame, stoi_predict_frame)
    print("Frame SDI vs Frame STOI: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(np.array(sdi_predict_frame).T, np.array(stoi_predict_frame).T)
    print("Frame SDI vs Frame STOI: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(sdi_predict_frame, stoi_predict_frame, s=14)
    plt.xlabel('Predicted SDI Frame scores')
    plt.ylabel('Predicted STOI Frame scores')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Frame_SDI_vs_STOI_MOSA_Net_Cross_Domain.png', dpi=150)

    print("-" * 65)
    #################################################################################
    # Plotting the scatter plot AVERAGE SDI vs PESQ
    LCC = np.corrcoef(sdi_predict, pesq_predict)
    print("Average SDI vs Average PESQ: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(sdi_predict.T, pesq_predict.T)
    print("Average SDI vs Average PESQ: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(sdi_predict, pesq_predict, s=14)
    plt.xlabel('Average Predicted SDI')
    plt.ylabel('Average Predicted PESQ')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Avg_SDI_vs_PESQ_MOSA_Net_Cross_Domain.png', dpi=150)

    # Plotting the scatter plot FRAME SDI vs PESQ
    LCC = np.corrcoef(sdi_predict_frame, pesq_predict_frame)
    print("Frame SDI vs Frame PESQ: Linear correlation coefficient: {:0.4f}".format(LCC[0][1]))
    SRCC = scipy.stats.spearmanr(np.array(sdi_predict_frame).T, np.array(pesq_predict_frame).T)
    print("Frame SDI vs Frame PESQ: Spearman rank correlation coefficient: {:0.4f}".format(SRCC[0]))
    plt.figure(1)
    plt.scatter(sdi_predict_frame, pesq_predict_frame, s=14)
    plt.xlabel('Predicted SDI Frame scores')
    plt.ylabel('Predicted PESQ Frame scores')
    plt.title("LCC: {:0.4f},  SRCC: {:0.4f}".format(LCC[0][1], SRCC[0]))
    plt.show()
    plt.savefig('Scatter_plot_Frame_SDI_vs_PESQ_MOSA_Net_Cross_Domain.png', dpi=150)


if __name__ == '__main__':
    pathmodel = "MOSA_Net_Cross_Domain_Multi_Target"
    cross_domain_inference(utils.input_list_read('../data/io_test_list.txt'),
                           utils.input_list_read('../data/List_Npy_Val_hubert_MOS_Challenge_phase1_main.txt'))
