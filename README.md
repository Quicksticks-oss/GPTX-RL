# GPTX-RL
### Features
- Just like GPT this model generates text based on an input prompt.
- This project uses Reinforcement Learning (RL) for training and inference.
- All models can be found on hugging face.

---

### Models
- [GPTX-RL-371K (Very Small) (Complete!)](https://huggingface.co/printr/gptx-rl-371k "GPTX-RL-371K ")


- [GPTX-RL-1M (Small) (Coming Soon!)](https://huggingface.co/printr/gptx-rl-371k "GPTX-RL-371K ")
- [GPTX-RL-14M (Medium) (Coming Soon!)](https://huggingface.co/printr/gptx-rl-371k "GPTX-RL-371K ")
- [GPTX-RL-140M (Large) (Coming Soon!)](https://huggingface.co/printr/gptx-rl-371k "GPTX-RL-371K ")
- [GPTX-RL-440M (Huge)  (Coming Soon!)](https://huggingface.co/printr/gptx-rl-371k "GPTX-RL-371K ")

---
### Training

- First step is to run ``generate_vocab.py`` on whatever text data you would like to train on for example ``python3 generate_vocab.py -i train.txt``
- Next step is to run ``train.py``
- Now all you have to do is wait!
---
### Inference

- not fully complete but inferect can be removed from ``train.py`` and put into its own script.

### Logging
- Output plot graphs will be avalible in the next version.
