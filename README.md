# PPDAI Magic Mirror Data Application Contest

## Introduction
This is the repository for [PPDAI](https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1&tabindex=2) contest, which is a natural language processing (NLP) model aims to detect duplicate questions in Chinese. 

## Data
Data was provided by PPDAI, which are pairs of questions labeled with 0 and 1 represents similar or not.
The questions are represented by two sequences of integers which are the indices of corresponding embedding vectors (word and character). 

## Model
We proposed three models including a RNN based model, CNN based model and a RCNN based model. These models have the following characteristics:  

1. Bi-Directional GRU in RNN based models for semantic learning.
2. 1-D Convolution in CNN and RCNN based models for local feature extraction.
3. Co-Attention was used to learn the semantic correlations between two sequences.
4. Self-Attention was used to enhance the feature representation.
5. Word embedding and Character Embedding were used simultaneously.

## Performance:
Our ensemble model achieved 0.203930 of loss in the semi-final, at the top 15% in ranking.

## Reference
[QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension[ICLR 2018]](https://arxiv.org/abs/1804.09541)  
  
[Zhouhan Lin et al. “A Structured Self-attentive Sentence Embedding”. In:CoRRabs/1703.03130 (2017).arXiv:1703.03130.](http://arxiv.org/abs/1703.03130.)  

[ Pranav Rajpurkar et al. “SQuAD: 100, 000+ Questions for Machine Comprehension of Text”. In:CoRRabs/1606.05250 (2016). arXiv:1606.05250.](http://arxiv.org/abs/1606.05250.)  

[Wenhui Wang et al. “Gated Self-Matching Networks for Reading Comprehension and Question Answering”](http://www.aclweb.org/anthology/P17-1018)  
