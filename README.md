# Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features

### Introduction ###

This fork of the [original MOSA-Net Cross Domain repo](https://github.com/dhimasryan/MOSA-Net-Cross-Domain), implements 
a number of improvements to evaluate the two pretrained models that predict [VoiceMOS](https://github.com/dhimasryan/MOSA-Net-Cross-Domain/tree/main/PreTrained_VoiceMOSChallenge) & [{PESQ, SDI, STOI}](https://github.com/dhimasryan/MOSA-Net-Cross-Domain/tree/main/PreTrained_WSJ).

The changes and light refactoring include
- Move `2.7` & `3.6` (fairseq) code to `python3.7`, modified to do inference on `cpu`
- Update strictly-required dependencies for inference, captured in a `requirements.txt`
- Move essential code to a new `src` directory 
- Move all pretrained models to `pretrained_models`
- Streamline evaluation of long files with the development of a simple CLI tool, which segments audio into utterances, 
before computing features (spectrogram, waveform, HuBERT) and doing inference, on each utterance, using both models.

### Installation ###
Tested under `python3.7`, install dependencies with
```
pip install -r requirements.txt
```

### How to run the code ###
`io_mosanet.py` and `io_mosanet_crossdomain.py` evaluate [VoiceMOS](https://github.com/dhimasryan/MOSA-Net-Cross-Domain/tree/main/PreTrained_VoiceMOSChallenge) & [{PESQ, SDI, STOI}](https://github.com/dhimasryan/MOSA-Net-Cross-Domain/tree/main/PreTrained_WSJ) respectively
using HuBERT features, specifically extracted using `io_extract_hubert.py`

To evaluate longer audio files (eg. 30min), the simple CLI tool will first VAD the audio file into utterances and then compute
all metrics on each utterance.
```
Usage: iorife_dialog_intel_cli.py [path-to-audio-file]
```

### Citation ###
Please kindly cite the original authors' paper

<a id="1"></a> 
R. E. Zezario, S. -W. Fu, F. Chen, C. -S. Fuh, H. -M. Wang and Y. Tsao, 
"Deep Learning-Based Non-Intrusive Multi-Objective Speech Assessment Model With Cross-Domain 
Features," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 54-70, 
2023, doi: 10.1109/TASLP.2022.3205757.

### Note ###

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a>, <a href="https://github.com/grausof/keras-sincnet" target="_blank">SincNet</a>, <a href="https://github.com/pytorch/fairseq" target="_blank">Self-Supervised Learning Model</a> are created by others
