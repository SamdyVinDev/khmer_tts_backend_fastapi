#!/usr/bin/env python
# coding: utf-8

tacotron2_model = "model/khmerTTS_male.pt"
waveglow_model = "model/waveglow_pretrained.pt"

import sys

# from os.path import join

# sys.path.append(join("tacotron2", "waveglow/"))
# sys.path.append("tacotron2")

import numpy as np
import torch

# import matplotlib.pylab as plt
from ml_model.tacotron2.hparams import create_hparams
from ml_model.tacotron2.model import Tacotron2
from ml_model.tacotron2.text import text_to_sequence
from ml_model.tacotron2.waveglow.denoiser import Denoiser

import warnings

warnings.filterwarnings("ignore")
import time

# for ploting data , no need for website
# def plot_data(data,current):
#     graph_width = 900
#     graph_height = 360
#     figsize=(int(graph_width/100), int(graph_height/100))
#     fig, axes = plt.subplots(1, len(data), figsize=figsize)
#     for i in range(len(data)):
#         axes[i].imshow(data[i], aspect='auto', origin='bottom',
#                        interpolation='none', cmap='inferno')
#     fig.canvas.draw()
#     # plt.show()
#     from os import path
#     if not path.exists('example/result/plot/'):
#         os.mkdir('example/result/plot/')
#     fig.savefig('example/result/plot/'+current+'.png')

curlist = {
    "$": "ដុល្លារ",
    "៛": "រៀល",
    "€": "អឺរ៉ូ",
    "¥": "យេន",
    "￥": "យន់",
    "₹": "រូពី",
    "£": "ផោន",
    "฿": "បាត",
    "₫": "ដុង",
    "₭": "គីប",
}

thisdict = {}


def ARPA(text):
    start = time.perf_counter()
    out = ""
    for word_ in text.split(" "):
        word = word_
        end_chars = ""

        import re
        from ml_model.text2num.text.num2word import num2word, num_en2km

        # convert space to , to separate sentense
        word = re.sub(r"\s", ",", word)
        # convert english to khmer
        word = re.sub(r"[0-9]+", num_en2km, word)
        # covert number to text

        # pattern = re.compile("^([A-Z][0-9]+)+$")
        # if pattern.match(string)

        word = re.sub(r"[+-]?([០-៩]*[,])?[០-៩]+", num2word, word)

        # convert some currency symbol to khmer text
        word = re.sub(r"[$៛€¥￥₹£฿₫₭]", lambda m: curlist.get(m.group()), word)

        while any(elem in word for elem in r"!?,.;។") and len(word) > 1:
            if word[-1] == "!":
                end_chars = "!" + end_chars
                word = word[:-1]
            if word[-1] == "?":
                end_chars = "?" + end_chars
                word = word[:-1]
            if word[-1] == ",":
                end_chars = "," + end_chars
                word = word[:-1]
            if word[-1] == ".":
                end_chars = "." + end_chars
                word = word[:-1]
            if word[-1] == ";":
                end_chars = ";" + end_chars
                word = word[:-1]
            if word[-1] == "។":
                end_chars = "។" + end_chars
                word = word[:-1]
        try:
            word_arpa = thisdict[word.upper()]
        except:
            word_arpa = ""
        if len(word_arpa) != 0:
            word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    # check if the output sentent has end_chars , if not add ។
    if not any(elem in out for elem in r"!?,.;។"):
        out = out + " ។"
    # make sure end_chars mush be ;
    if out[-1] != ";":
        out = out + ";"
    duration = time.perf_counter() - start
    print("ARPA,Duration {:.6f}".format(duration))
    return out


# load model tacontron 2
def loadTaco(path, device):
    start = time.perf_counter()
    # initialize Tacotron2 with the pretrained model
    hparams = create_hparams()
    hparams.sampling_rate = 22050  # Don't change this
    hparams.max_decoder_steps = 10000  # How long the audio will be before it cuts off (1000 is about 11 seconds),in here we set to 10000 which almost 2 minutes
    hparams.gate_threshold = 0.1  # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
    # loading model
    tacotron = Tacotron2(hparams)
    if device == "gpu":
        tacotron.load_state_dict(torch.load(path))
        _ = tacotron.cuda().eval().half()
    elif device == "cpu":
        tacotron.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        _ = tacotron.eval()
    duration = time.perf_counter() - start
    print("loadTaco,Duration {:.6f}".format(duration))
    return tacotron


# load waveplow
def loadWave(path, device):
    start = time.perf_counter()
    if device == "gpu":
        waveglow = torch.load(path)["model"]
        waveglow.cuda().eval().half()
        # denoiser = Denoiser(waveglow)
    elif device == "cpu":
        waveglow = torch.load(path, map_location=torch.device("cpu"))["model"]
        waveglow.eval()
    for k in waveglow.convinv:
        k.float()
    duration = time.perf_counter() - start
    print("loadWave,Duration {:.6f}".format(duration))
    return waveglow


# text = ""
# with open('example/text.txt') as f:
#     text = f.read()
#     f.close()

# function for convert text to sound
def synthesis(text, tacotron, waveglow, path, device):
    sigma = 0.8
    denoise_strength = 0.324
    raw_input = False  # disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing

    for input in text.split("\n"):
        start = time.perf_counter()
        if len(input) < 1:
            continue
        print(input)
        if raw_input:
            if input[-1] != ";":
                input = input + ";"
        else:
            input = ARPA(input)
        print(input)

        with torch.no_grad():  # save VRAM by not including gradients
            sequence = np.array(
                text_to_sequence(input, ["english_cleaners"]), dtype=np.float32
            )[None, :]

            start1 = time.perf_counter()
            if device == "gpu":
                sequence = (
                    torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                )
            elif device == "cpu":
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
            duration1 = time.perf_counter() - start1
            print("Load input,Duration {:.6f}".format(duration1))

            startaco = time.perf_counter()
            mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(
                sequence
            )
            endtaco = time.perf_counter() - startaco
            print("tacotron.inference,Duration {:.6f}".format(endtaco))

            ##just plot data
            # plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],alignments.float().data.cpu().numpy()[0].T),current)
            # ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), 22050))
            # import os
            # from os import path
            # if not path.exists('example/result/sound/'):
            #     os.mkdir('example/result/sound/')

            startwave = time.perf_counter()
            audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)
            endwave = time.perf_counter() - startwave
            print("waveglow.inference,Duration {:.6f}".format(endwave))

            print("")

            from scipy.io.wavfile import write

            # current = str(int(time.time()))
            write(path, 22050, audio[0].data.cpu().numpy().astype(np.float32))
            duration = time.perf_counter() - start
            print("")
            # print("sound file saved at ",'example/result/sound/'+current+'.wav')
            print("Complete,Duration {:.6f}".format(duration))


tacoCpu = loadTaco(tacotron2_model, "cpu")
tacoGpu = loadTaco(tacotron2_model, "gpu")
waveCpu = loadWave(waveglow_model, "cpu")
waveGpu = loadWave(waveglow_model, "gpu")

# "តម្លៃ​សាំង​លក់ រាយ​ នៅ​ទីផ្សារ​កម្ពុជា​ធ្លាក់ចុះ​ ៥០ ​រៀល​ក្នុង​ ១​ លីត្រ។"
def Run(text, path, device="cpu"):
    if device == "gpu":
        synthesis(text, tacoGpu, waveGpu, path, "gpu")
    elif device == "cpu":
        synthesis(text, tacoCpu, waveCpu, path, "cpu")


# Run(
#     "ស្ថិតនៅចំណុចប្រសព្វគ្នានៃ ទន្លេមេគង្គ ទន្លេបាសាក់ និង ទន្លេសាប។ ភ្នំពេញគឺ ជាទីក្រុងធំនៃប្រទេសកម្ពុជាហើយមានប្រជាជនជាងពីរលាននា",
#     "/home/lyhourt.te/source/synthesis/example/result/sound/test1.wav",
#     "cpu",
# )
