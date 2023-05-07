# BIAS-Bias-In-AI-S
In this notebook we will have a look how one can see how a model is biased with just few lines of code. 
First, we need to recap what Transformers are and how they work. This is a very short introduction to Transformers, if you want to know more about them, I recommend you to read the [original paper](https://arxiv.org/pdf/1706.03762.pdf) or [this blog post](https://jalammar.github.io/illustrated-transformer/).

In this session, our focus will be on utilizing BERT for language modeling and detecting the various types of biases that the model might have learned from the training data.

Transformers are a type of deep learning model used primarily for natural language processing (NLP) tasks, such as language translation and sentiment analysis. They were introduced in a 2017 paper by Vaswani et al. and have become increasingly popular due to their high accuracy and ability to handle long-range dependencies in data.

The key feature of Transformers is their attention mechanism, which allows them to focus on different parts of the input data when making predictions. This attention mechanism is based on a set of "queries," "keys," and "values" that are learned during training. The queries are used to determine which parts of the input data are most relevant for a particular prediction, and the keys and values are used to calculate a weighted sum that represents this relevance.

To make a prediction, a Transformer takes in an input sequence of tokens (such as words or characters) and processes them through a series of "encoder" and "decoder" layers. The encoder layers are used to generate a representation of the input sequence, while the decoder layers use this representation to generate the output sequence. The output sequence is generated one token at a time, with each token being predicted based on the previous tokens and the input sequence.

MASK is a specific task that can be performed with Transformers, in which a portion of the input sequence is randomly "masked" (replaced with a special token) and the Transformer is trained to predict the original value of these masked tokens. This task is useful for pre-training Transformers on large amounts of data, as it encourages the model to learn representations that are robust to missing or incomplete input. MASK is used in the pre-training stage of models such as BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa, which have achieved state-of-the-art performance on a wide range of NLP tasks.

When we give a "masked" sentence to BERT, which is a type of Transformer-based model that uses the MASK task during pre-training, BERT will predict the original value of the masked token based on the context of the surrounding tokens in the sentence.

For example, suppose we have the sentence: "The cat sat on the [MASK]." If we give this sentence to BERT, it will predict the most likely word to fill in the masked position based on its understanding of the meaning of the sentence and the distribution of words in the training data. BERT generates a probability distribution over the entire vocabulary for the masked token, and the predicted word is the one with the highest probability.

The MASK task is used during pre-training to help BERT learn contextualized word embeddings, which are representations of words that capture their meaning in context. By predicting the masked tokens, BERT learns to understand how words relate to each other in a sentence and how their meaning can change based on the context in which they are used. This knowledge can then be applied to a wide range of downstream NLP tasks, such as sentiment analysis, question answering, and language translation.

We will use this trick and get the predictions of the model for a given sentence. Then we will mask a word and see what are the prediction. Let's see how we can do this.

```
1. He is a good footbal player. The word 'He' is predicted with a probability of 60.0%
2. She is a good footbal player. The word 'She' is predicted with a probability of 8.95%
3. he is a good footbal player. The word 'he' is predicted with a probability of 0.68%
4. . is a good footbal player. The word '.' is predicted with a probability of 0.59%
5. It is a good footbal player. The word 'It' is predicted with a probability of 0.28%
6. David is a good footbal player. The word 'David' is predicted with a probability of 0.2%
7. John is a good footbal player. The word 'John' is predicted with a probability of 0.19%
8. Michael is a good footbal player. The word 'Michael' is predicted with a probability of 0.18%
9. Joe is a good footbal player. The word 'Joe' is predicted with a probability of 0.15%
10. and is a good footbal player. The word 'and' is predicted with a probability of 0.14%
```
![Top 10 words for the masked word](https://user-images.githubusercontent.com/27974341/236682249-0617b42c-e1ef-4706-818e-4b3b5b01313a.svg)



