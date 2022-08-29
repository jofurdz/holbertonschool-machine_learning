#!/usr/bin/env python3
"""module containing function gensim_to_keras"""
from gensim.models import Word2Vec
import gensim.models


def gensim_to_keras(model):
    """converts word2vec model to a keras embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=True)
