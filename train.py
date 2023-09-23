import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

from preprocess import preprocess as p
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

if __name__ == '__main__':
    data = p()
    result = data.preprocess()
    model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

    model.wv.save_word2vec_format('./ckpt/eng_w2v')  # 모델 저장

