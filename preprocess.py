import re
import urllib.request
import zipfile
import nltk

from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

class preprocess:
    def __init__(self):
        self.targetXML = open('./data/ted_en-20160408.xml', 'r', encoding='UTF8')
        self.target_text = etree.parse(self.targetXML)

    def preprocess(self):
        # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
        parse_text = '\n'.join(self.target_text.xpath('//content/text()'))

        # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
        # 해당 코드는 괄호로 구성된 내용을 제거.
        content_text = re.sub(r'\([^)]*\)', '', parse_text)

        # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
        print('tokenize start..')
        sent_text = sent_tokenize(content_text)
        print('tokenize end..')
        # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
        normalized_text = []
        for string in sent_text:
             tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
             normalized_text.append(tokens)

        # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
        result = [word_tokenize(sentence) for sentence in normalized_text]
        return result