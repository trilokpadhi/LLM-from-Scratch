This repository is a motivation to implement LLMs from Scratch using PyTorch. The repository is inspired by the [Hugging Face Transformers](), [The Annotated Transformer](), and [The Illustrated Transformer]() repositories. The goal is to understand the architecture and the implementation details of the LLMs. The repository is a work in progress and will be updated regularly.


## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Implementation](#implementation)
- [Usage](#usage)
- [References](#references)
- [License](#license)

## Introduction
The LLMs are a type of neural network that is trained to predict the next word in a sentence given the previous words. The LLMs are used in various NLP tasks such as text generation, machine translation, and sentiment analysis. The LLMs are based on the Transformer architecture, which is a type of neural network that uses self-attention mechanism to process the input sequence. It is composed of an encoder and a decoder, which are used to process the input sequence and generate the output sequence, respectively. The LLMs are trained using a large corpus of text data, which is used to learn the patterns in the text data and generate the output sequence.

## Architecture
The LLMs are based on the Transformer architecture, which is composed of an encoder and a decoder, which are used to process the input sequence and generate the output sequence, respectively. The encoder is used to process the input sequence and generate a representation of the input sequence, which is used by the decoder to generate the output sequence. The encoder and decoder are composed of multiple layers of self-attention mechanism, which is used to process the input sequence and generate the output sequence. The self-attention mechanism is used to compute the attention weights between the input sequence and the output sequence, which are used to generate the output sequence.

## Implementation
The implementation of the LLMs is done using PyTorch, which is a popular deep learning library in Python. The implementation is based on the Transformer architecture, which is a type of neural network that uses self-attention mechanism to process the input sequence. The implementation is done in a modular way, which allows for easy customization of the architecture and the training process. The implementation is inspired by the [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), and [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), which provide a detailed explanation of the Transformer architecture and its implementation details.

## Usage
The repository contains the implementation of the LLMs in PyTorch. The implementation is done in a modular way, which allows for easy customization of the architecture and the training process. The repository contains the following files:

- `transformer.py`: Contains the implementation of the Transformer architecture.
- `train.py`: Contains the training script for the LLMs.
- `generate.py`: Contains the generation script for the LLMs.

To train the LLMs, run the following command:
```bash
python train.py
```

To generate text using the trained LLMs, run the following command:
```bash
python generate.py
```

## References
### Blogs
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [A visual deep dive into transformers architecture](https://francescopochetti.com/a-visual-deep-dive-into-the-transformers-architecture-turning-karpathys-masterclass-into-pictures/)
- [The Transformer Family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)
- [Transformers from Scratch by Brandon Rohrer](https://e2eml.school/transformers)
- [TRANSFORMERS FROM SCRATCH](https://peterbloem.nl/blog/transformers)[code](https://github.com/pbloem/former)[video](https://www.youtube.com/playlist?list=PLIXJ-Sacf8u60G1TwcznBmK6rEL3gmZmV)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
### Repositories
- [Hugging Face Transformers]()
- [The Annotated Transformer]()
- [The Illustrated Transformer]()
- [GPT-3: Language Models are Few-Shot Learners]()
- [GPT-practice](https://github.com/PetropoulakisPanagiotis/gpt-practice/tree/main)
### Courses
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Stanford CS231N: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Stanford CS229: Machine Learning](http://cs229.stanford.edu/)
- [Stanford CS230: Deep Learning](http://cs230.stanford.edu/)   

