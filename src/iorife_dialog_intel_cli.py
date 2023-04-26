
import torch
import time
import os
import runez
import sys
import json
import numpy as np
import glob
from os.path import join
import io_mosanet
import io_mosanet_cross_domain
import fairseq
import librosa
import shutil
import utils

torch.set_num_threads(1)
SAMPLING_RATE = 16000
temp_output_dir = "/tmp/silero_vad_segments/"

start_time = time.time()
model, vad_utils = torch.hub.load(repo_or_dir='/Users/iorife/github/silero-vad/',
                              model='silero_vad',
                              source='local',
                              force_reload=True,
                              onnx=False)

print("[VAD] model loading --- %s seconds ---" % (time.time() - start_time))
(get_speech_timestamps, _, read_audio, _, _) = vad_utils


def vad_audio(path):
    start_time = time.time()
    print("[VAD] segmenting {} ".format(path))
    wav = read_audio(path, sampling_rate=SAMPLING_RATE)
    # get number timestamps from full audio file
    vad_timestamps = get_speech_timestamps(wav, model, min_speech_duration_ms=2000, min_silence_duration_ms=300)
    print("[VAD] predictions --- %s seconds ---" % (time.time() - start_time))
    print("[VAD] segments: {}".format(len(vad_timestamps)))
    # pprint(vad_timestamps)
    return vad_timestamps


def segment_original_audiofile(timestamps, single_channel_wav_path):
    # clean up previous runs
    try:
        shutil.rmtree(temp_output_dir)
    except OSError as e:
        print("rmtree Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    print("[VAD] snipping (ffmpeg)")
    # Generate WAVS for each utterance
    start_time = time.time()
    index = 0
    for segment in timestamps:
        start = segment['start' ]/ SAMPLING_RATE  # need seconds
        end = segment['end'] / SAMPLING_RATE  # need seconds
        duration = end - start
        segment_file, ext = os.path.splitext(os.path.basename(single_channel_wav_path))

        # check for empty extension
        if not ext:
            ext = ".wav"
        segment_file = str(index) + "_" + segment_file + "_" + str(start) + "_" + str(end) + ext
        segment_fullpath = os.path.join(temp_output_dir, segment_file)
        print(".", end="")
        run_result = runez.run(
            runez.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "panic",
            "-ss",
            start,
            "-t",
            duration,
            "-i",
            "%s" % single_channel_wav_path,
            "-ar",
            SAMPLING_RATE,  # doing SRC now, during utterance segmentation will save CPU during inference
            segment_fullpath,
            fatal=False,
        )
        if run_result.failed:
            print("Error ffmpeg segment file failed")

        index += 1
    print("\n[VAD] snipping took --- %s seconds ---" % (time.time() - start_time))


def do_dialog_intel_inference():
    # Build list of utterances
    utt_file_list = glob.glob(join(temp_output_dir, "**/*.wav"), recursive=True)
    num_vad_segments = len(utt_file_list)
    print("[VAD] Number of utts segmented: {} ".format(num_vad_segments))

    start_time = time.time()
    print("\n[Dialog Intel] loading the VoiceMOS MOSA-Net model ...")
    voicemos_model = io_mosanet.BLSTM_CNN_with_ATT_VoiceMOS()
    voicemos_model.load_weights('../pretrained_models/MOSA-Net_Cross_Domain_epoch_100.h5')
    print(voicemos_model.summary())

    print("\n[Dialog Intel] loading the {PESQ, SDI, STOI} MOSA-Net model ...")
    crossdomain_model = io_mosanet_cross_domain.BLSTM_CNN_with_ATT_cross_domain()
    crossdomain_model.load_weights('../pretrained_models/MOSA_Net_Cross_Domain_Multi_Target.h5')
    print(crossdomain_model.summary())

    print("\n[Dialog Intel] loading the Hubert SSL model ...")
    cp_path = '../pretrained_models/hubert_large_ll60k.pt'
    hubert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    hubert_model = hubert_model[0]
    hubert_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hubert_model = hubert_model.to(device)

    for i in range(num_vad_segments):
        utterance_file = utt_file_list[i]

        # get base-filename, remove extension
        basename_list = os.path.splitext(os.path.basename(utterance_file))[0].split("_")
        start, end = float(basename_list[-2]), float(basename_list[-1])

        # setup event
        hundred_nano_mulitplier = 10000000.0
        event = {'endTime': int(end * hundred_nano_mulitplier),
                 'startTime': int(start * hundred_nano_mulitplier),
                 'metadata': {}}

        # compute Hubert
        audio_data, _ = librosa.load(utterance_file, sr=16000, mono=True)
        end2end_channel_1 = np.reshape(audio_data, (1, audio_data.shape[0]))
        end2end_channel_1 = torch.from_numpy(end2end_channel_1).to("cpu")
        features_1 = hubert_model(end2end_channel_1, features_only=True, mask=False)['x']
        causal_1 = features_1.detach().to("cpu").numpy()

        # ------------------------------------------------------------
        # VoiceMOS Model prediction
        noisy_spectrogram, noisy_waveform = utils.spectrogram_and_phase(utterance_file)
        noisy_hubert = causal_1
        [voice_mos_score, frame_mos] = voicemos_model.predict([noisy_spectrogram, noisy_waveform, noisy_hubert],
                                                              verbose=0, batch_size=1)
        denorm_mos_predict = utils.denorm(voice_mos_score)
        estimated_score = denorm_mos_predict[0]

        # ------------------------------------------------------------
        # Cross Domain Model prediction
        [PESQ_score, frame_score, STOI_score, frame_stoi, SDI_score, frame_sdi] = crossdomain_model.predict(
            [noisy_spectrogram, noisy_waveform, noisy_hubert], verbose=0, batch_size=1)

        # PESQ_score, SDI_score, STOI_score = None, None, None
        print("file: {}\tVoiceMOS: {}\tPESQ: {}\tSDI: {}\tSTOI: {}".format(os.path.basename(utterance_file),
                                                                                estimated_score,
                                                                                PESQ_score[0],
                                                                                SDI_score[0],
                                                                                STOI_score[0]))
    print("Model loading and inference took --- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: iorife_dialog_intel_cli.py [path-to-audio-file]\n")
        exit()

    asset_path = sys.argv[1]
    timestamps = vad_audio(asset_path)
    segment_original_audiofile(timestamps, asset_path)
    do_dialog_intel_inference()