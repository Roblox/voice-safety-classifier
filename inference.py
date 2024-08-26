# Copyright © 2024 Roblox Corporation

"""
This file gives a sample demonstration of how to use the given functions in Python, for the Voice Safety Classifier model. 
"""

import torch
import libro
import numpy as np
import argparse
from transformers import WavLMForSequenceClassification


def feature_extract_simple(
    wav,
    sr=16_000,
    win_len=15.0,
    win_stride=15.0,
    do_normalize=False,
):
    """simple feature extraction for wavLM
    Parameters
    ----------
    wav : str or array-like
        path to the wav file, or array-like
    sr : int, optional
        sample rate, by default 16_000
    win_len : float, optional
        window length, by default 15.0
    win_stride : float, optional
        window stride, by default 15.0
    do_normalize: bool, optional
        whether to normalize the input, by default False.
    Returns
    -------
    np.ndarray
        batched input to wavLM
    """
    if type(wav) == str:
        signal, _ = librosa.core.load(wav, sr=sr)
    else:
        try:
            signal = np.array(wav).squeeze()
        except Exception as e:
            print(e)
            raise RuntimeError
    batched_input = []
    stride = int(win_stride * sr)
    l = int(win_len * sr)
    if len(signal) / sr > win_len:
        for i in range(0, len(signal), stride):
            if i + int(win_len * sr) > len(signal):
                # padding the last chunk to make it the same length as others
                chunked = np.pad(signal[i:], (0, l - len(signal[i:])))
            else:
                chunked = signal[i : i + l]
            if do_normalize:
                chunked = (chunked - np.mean(chunked)) / (np.std(chunked) + 1e-7)
            batched_input.append(chunked)
            if i + int(win_len * sr) > len(signal):
                break
    else:
        if do_normalize:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-7)
        batched_input.append(signal)
    return np.stack(batched_input)  # [N, T]


def infer(model, inputs):
    output = model(inputs)
    probs = torch.sigmoid(torch.Tensor(output.logits))
    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        help="File to run inference",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="roblox/voice-safety-classifier",
        help="checkpoint file of model",
    )
    args = parser.parse_args()
    labels_name_list = [
        "Profanity",
        "DatingAndSexting",
        "Racist",
        "Bullying",
        "Other",
        "NoViolation",
    ]
    # Model is trained on only 16kHz audio
    audio, _ = librosa.core.load(args.audio_file, sr=16000)
    input_np = feature_extract_simple(audio, sr=16000)
    input_pt = torch.Tensor(input_np)
    model = WavLMForSequenceClassification.from_pretrained(
        args.model_path, num_labels=len(labels_name_list)
    )
    probs = infer(model, input_pt)
    probs = probs.reshape(-1, 6).detach().tolist()
    print(f"Probabilities for {args.audio_file} is:")
    for chunk_idx in range(len(probs)):
        print(f"\nSegment {chunk_idx}:")
        for label_idx, label in enumerate(labels_name_list):
            print(f"{label} : {probs[chunk_idx][label_idx]}")
