## Model Description
We present a voice-safety classification model that can be used for voice-toxicity detection and classification.
The model has been distilled into the [WavLM](https://arxiv.org/abs/2110.13900) architecture from a larger teacher model.
All the model training has been conducted with Roblox internal voice chat datasets,
using both machine and human-labeled data, with over 100k hours of training data in total.
We have also published a blog post about this work.

The model supports eight languages: English, Spanish, German, French, Portuguese, Italian, Korean, and Japanese.
It classifies the input audio into six toxicity classes in a multilabel fashion. The class labels are as follows:
`Discrimination`, `Harassment`, `Sexual`, `IllegalAndRegulated`, `DatingAndRomantic`, and `Profanity`.
Please refer to [Roblox Community Standards](https://en.help.roblox.com/hc/en-us/articles/203313410-Roblox-Community-Standards)
for a detailed explanation on the policy, which has been used for labeling the datasets.
The model outputs have been calibrated for the Roblox voice chat environment,
so that the class scores after a sigmoid can be interpreted as probabilities.

The classifier expects 16kHz audio segments as input. Ideal segment length is 15 seconds,
but the classifier can operate on shorter segments as well. The prediction accuracy may degrade
for longer segments.

The table below displays evaluation precision and recall for each of the supported languages,
as calculated over internal language-specific held-out datasets, which resemble the Roblox voice chat traffic.
The operating thresholds for each of the categories were kept equal per language, and optimized
to achieve a false positive rate of 1%. The classifier was then evaluated as a binary classifier,
tagging the audio as positive if any of the heads exceeded the threshold.

|Language|Precision|Recall|
|---|---|---|
|English   |63.9%|58.2%|
|Spanish   |76.1%|63.2%|
|German    |69.9%|74.1%|
|French    |70.3%|69.8%|
|Portuguese|85.4%|58.0%|
|Italian   |86.6%|52.4%|
|Korean    |78.0%|64.6%|
|Japanese  |56.7%|57.7%|

Compared to the v1 voice safety classifier, the v2 model
expands the support from English to 7 additional languages,
as well as significantly improving the classification accuracy.
With the 1% false positive rate as above, the binary recall for English is improved by 92%.


## Usage
The dependencies for the inference file can be installed as follows:
```
pip install -r requirements.txt
```
The provided Python file demonstrates how to use the classifier with arbitrary 16kHz audio input.
To run the inference, please run the following command:
```
python inference.py --audio_file <your audio file path> --model_path <path to Huggingface model>
```
You can download the model weights from the model releases page [here](https://github.com/Roblox/voice-safety-classifier/releases/tag/vs-classifier-v2),
or from HuggingFace under [`roblox/voice-safety-classifier-v2`](https://huggingface.co/Roblox/voice-safety-classifier-v2).
If `model_path` isn’t specified, the model will be loaded directly from HuggingFace.
