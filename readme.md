## Finetuning CTC models on other languages

- **How to fine-tune models on low-resource languages efficiently?**

### Sub-word Encoding CTC Model

A sub-encoding model accepts a sub-word tokenized text corpus and emits sub-word tokens in its decoding step. 
This repository will detail how we prepare a CTC model which utilizes a sub-word Encoding scheme.

We will utilize a pre-trained Citrinet model trained on roughly 7,000 hours of English speech as the base model. 
We will modify the decoder layer (thereby changing the model's vocabulary) for training.

#### Load Citrinet model

<img src="citrinet_model_params.png" width="340" height="141">

#### Referans
[Jump-start Training for Speech Recognition Models in Different Languages with NVIDIA NeMo](https://developer.nvidia.com/blog/jump-start-training-for-speech-recognition-models-with-nemo/)
