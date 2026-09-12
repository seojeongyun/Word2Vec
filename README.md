# Word2Vec

## Overview

자연어처리(NLP)를 처음 공부하며 **텍스트 전처리, Word Embedding, Word2Vec의 기본 개념과 사용 방법을 실습한 프로젝트**입니다.

영어 Corpus를 대상으로 텍스트를 전처리한 뒤 Gensim의 Word2Vec을 이용해 단어 임베딩을 학습하고, 영어 및 한국어 Word2Vec 실습을 통해 단어 간 의미적 관계를 확인했습니다.

본 Repository는 완성된 NLP Application보다는 **자연어처리의 기본 개념과 Word Embedding을 학습하기 위한 Study Project**에 가깝습니다.

---

## Pipeline

```text
Raw Text Corpus
      │
      ▼
Text Preprocessing
      │
      ├── Text Cleaning
      ├── Lowercasing
      ├── Sentence Tokenization
      └── Word Tokenization
      │
      ▼
Tokenized Sentences
      │
      ▼
Word2Vec Training
      │
      ▼
Word Embeddings
      │
      ▼
Word Similarity / Representation
```

---

## Text Preprocessing

영어 Word2Vec 학습에서는 TED Talk Corpus를 사용합니다.

`data_download.py`에서 TED XML 데이터를 다운로드하고, `preprocess.py`에서 Word2Vec 학습에 사용할 수 있도록 전처리를 수행합니다.

주요 전처리 과정은 다음과 같습니다.

```text
TED XML Corpus
      │
      ▼
Extract <content>
      │
      ▼
Remove Background Text
(Audio), (Laughter), ...
      │
      ▼
Sentence Tokenization
      │
      ▼
Lowercase & Remove Punctuation
      │
      ▼
Word Tokenization
```

NLTK를 이용해 문장 및 단어 단위 Tokenization을 수행하며, 정규표현식을 활용하여 불필요한 문자열을 제거합니다.

---

## Word2Vec

Word2Vec은 단어를 고정된 길이의 Vector로 표현하여, 단어 사이의 의미적 관계를 Vector Space에서 표현하는 Word Embedding 기법입니다.

본 프로젝트에서는 **Gensim Word2Vec**을 활용하여 전처리된 영어 Corpus로 Word Embedding을 학습합니다.

```python
model = Word2Vec(
    sentences=result,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)
```

주요 설정은 다음과 같습니다.

| Parameter | Value | Description |
|---|---:|---|
| `vector_size` | 100 | Word Embedding Dimension |
| `window` | 5 | Context Window Size |
| `min_count` | 5 | Minimum Word Frequency |
| `workers` | 4 | Number of Worker Threads |
| `sg` | 0 | CBOW |

학습된 Word Vector는 다음 경로에 저장합니다.

```text
./ckpt/eng_w2v
```

---

## Word Embedding

Word Embedding을 통해 각각의 단어를 고차원 Vector로 표현할 수 있습니다.

```text
Word
 │
 ▼
Word2Vec
 │
 ▼
Embedding Vector

"king"  → [0.13, -0.42, ..., 0.27]
"queen" → [0.10, -0.38, ..., 0.31]
```

비슷한 문맥에서 사용되는 단어는 Embedding Space에서도 상대적으로 가까운 위치에 표현될 수 있습니다.

이를 통해 단어 간 유사도 분석이나 다양한 NLP Task의 입력 표현으로 활용할 수 있습니다.

---

## English / Korean Word2Vec Practice

Repository에는 영어와 한국어 Word2Vec을 직접 실습하기 위한 Jupyter Notebook이 포함되어 있습니다.

```text
Word2Vec_from_Gensim.ipynb
word2vec_for_korean.ipynb
```

이를 통해 Gensim 기반 Word2Vec의 사용 방법과 영어/한국어 Word Embedding 과정을 학습했습니다.

---

## Repository Structure

```text
Word2Vec/
│
├── README.md
│
├── data_download.py
│   └── TED Corpus Download
│
├── preprocess.py
│   └── Text Cleaning & Tokenization
│
├── train.py
│   └── Gensim Word2Vec Training
│
├── test.py
│   └── Word2Vec Test
│
├── Word2Vec_from_Gensim.ipynb
│   └── Gensim Word2Vec Practice
│
└── word2vec_for_korean.ipynb
    └── Korean Word2Vec Practice
```

---

## Key Features

* **NLP Basic Study**  
  자연어처리의 기본 개념과 Text Preprocessing 과정을 학습

* **Text Preprocessing**  
  정규표현식과 NLTK를 활용해 Corpus Cleaning 및 Tokenization 수행

* **Word2Vec Training**  
  Gensim을 이용해 영어 Corpus 기반 Word Embedding 학습

* **CBOW-based Embedding**  
  Context Window를 기반으로 단어의 의미적 Representation 학습

* **English / Korean Practice**  
  영어 및 한국어 Word2Vec 실습을 통해 Word Embedding 개념 학습

---

## Study Notes & References

기존 README에서 정리했던 자연어처리 학습 자료입니다.

### 1. 자연어처리 개요

* 텍스트 전처리  
  https://wikidocs.net/21694  
  토큰화, 정제/정규화, 어간추출, 불용어, 정규표현식, 정수 인코딩, 패딩, 원-핫 인코딩, 데이터 분리, 한국어 전처리

* 언어모델  
  https://wikidocs.net/21695  
  언어모델, 통계적 언어모델, N-gram 언어모델 등

* 카운트 기반 단어 표현  
  https://wikidocs.net/24557  
  Bag-of-Words, 문서-단어 행렬, TF-IDF

### 2. 워드 임베딩

* 워드 임베딩  
  https://wikidocs.net/33520

* Word2Vec  
  https://wikidocs.net/22660

### 3. RNN

* RNN 개념  
  https://wikidocs.net/48558

### 4. 자연어처리 Task

* 이름 분류  
  https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

* 자동 번역 (Seq2Seq)  
  https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

* 영어 / 한국어 Word2Vec 실습  
  https://wikidocs.net/50739

---

## Notes

이 Repository는 **자연어처리를 처음 공부하면서 만든 학습용 프로젝트**입니다.

Tokenization, Text Cleaning, Normalization, Word Embedding 등 자연어처리의 기본적인 전처리 과정을 학습하고, Word2Vec을 이용해 단어를 Vector로 표현하는 방법을 실습했습니다.

---

## Tech Stack

`Python` · `Gensim` · `NLTK` · `Word2Vec` · `NLP` · `Jupyter Notebook`
