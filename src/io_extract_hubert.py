"""
@author: Ryandhimas Zezario, iroro
ryandhimas@citi.sinica.edu.tw
"""

import argparse
import os

import fairseq
import librosa
import numpy as np
import torch
import utils


def extract_ssl_feat(filepath, model, list_new, dirname):
    name = filepath[0]
    name_without_ext = name[:-4]
    new_name = name_without_ext + '.npy'
    path = '../data/wav/' + name

    cached_path = dirname + str(new_name)
    audio_data, _ = librosa.load(path, sr=16000, mono=True)

    end2end_channel_1 = np.reshape(audio_data, (1, audio_data.shape[0]))
    end2end_channel_1 = torch.from_numpy(end2end_channel_1).to("cpu")
    features_1 = model(end2end_channel_1, features_only=True, mask=False)['x']
    causal_1 = features_1.detach().to("cpu").numpy()
    np.save(cached_path, causal_1)
    info = filepath[0] + ',' + str(cached_path)
    list_new.append(info)
    return list_new


def test_data_generator(file_list, model):
    dirname = '../data/Hubert/test/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('Creating Directory')

    list_new = []
    print('Extracting Val-SSL Features')
    for index in range(len(file_list)):
        pesq_filepath = file_list[index].split(',')
        list_new = extract_ssl_feat(pesq_filepath, model, list_new, dirname)

    with open('../data/uttlist_npy_hubert_features.txt', 'w') as g:
        for item in list_new:
            g.write("%s\n" % item)


def compute_hubert_features_save_npy(Val_data):
    cp_path = '../pretrained_models/hubert_large_ll60k.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_data_generator(Val_data, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    args = parser.parse_args()
    Val_list = utils.input_list_read('../data/io_test_list.txt')
    compute_hubert_features_save_npy(Val_list)
