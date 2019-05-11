# Skipgram-word2vec
Skipgram word2vec implementation in C.
It implements negative sampling(to improve over vanilla softmax layer as it would be super time-consuming), and uses unigram probability distribution to pick those "negative samples", also uses subsampling(basic probability frequency based) to reduce corpus size.

### Datasets to train this network
* text8 dataset
* 1-billion-word-language-modeling-benchmark-r13output dataset
