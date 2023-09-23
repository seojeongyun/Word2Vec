import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

from preprocess import preprocess as p
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

if __name__ == '__main__':
    loaded_model = KeyedVectors.load_word2vec_format("./ckpt/eng_w2v")  # 모델 로드
    model_result = loaded_model.most_similar("man")
    print(model_result)
