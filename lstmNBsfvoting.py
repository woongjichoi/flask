import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import warnings 
warnings.filterwarnings(action='ignore')
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
!pip install install h5py==2.10.0
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
!set -x \
&& pip install konlpy \
&& curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x
import pickle
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import konlpy
import io
import os
def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


//인텐트 종류
labels = ['주문 인텐트', '햄버거메뉴안내 인텐트', '음료메뉴안내 인텐트','주문확인 인텐트',
          '취소 인텐트','상품취소 인텐트','상품 변경 인텐트','수량변경 인텐트','매장포장 인텐트',
          '계산 인텐트','포인트 적립 인텐트','카드 인텐트','모바일 인텐트','음성인식/키오스크 사용방법 인텐트',
          '매장 안내 인텐트','세트메뉴 안내/주문 인텐트','사이드 메뉴 안내/주문 인텐트','행사 인텐트','반응 인텐트',
          '중간의도 인텐트','토핑 인텐트','가격 안내 인텐트','미매칭']

//인텐트 학습데이터셋 업로드
from google.colab import files
train_data = files.upload()

//1.데이터셋 정제 
//train에 학습데이터 저장
train = pd.read_csv(io.BytesIO(train_data['Intent_bilstm.csv']),encoding='cp949')
train


//muti-NB용 데이터
x_data = np.array([x for x in train['text']])
y_data = np.array([x for x in train['intent']])


//bi-lstm 모델의 학습데이터셋 vectorization

//bi-lstm용 데이터
X_train = np.array([x for x in train['text']])
Y_train = np.array([x for x in train['intent']])

print(X_train.shape)
print(Y_train.shape)

from keras.preprocessing.text import Tokenizer
vocab_size = 2000  
tokenizer = Tokenizer(num_words = vocab_size)  
tokenizer.fit_on_texts(X_train) # Tokenizer 에 학습 데이터입력
sequences_train = tokenizer.texts_to_sequences(X_train)    # 문장 내 모든 단어를 시퀀스 번호로 변환
print(len(sequences_train))

//bi-lstm 모델의 학습데이터셋 Embedding

#Embeding 하기
word_index = tokenizer.word_index #시퀀스 번호 사용하기
max_length =15    # 일단 단어 길이 15개정도로 정함
padding_type='post' # 빈 공간 padding하기 

train_x = pad_sequences(sequences_train, padding='post', maxlen=max_length)
train_y = tf.keras.utils.to_categorical(Y_train) # Y_train 에 원-핫 인코딩  

print(train_y)
print(train_y.shape)

//2.모델 구축

//Multinomial NB

//MutinomailNB,TF-IDF 모듈 호출
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import MultinomialNB
mecab = konlpy.tag.Mecab()

//데이터전처리
//특수기호 제거,형태소 분석, 불용어 제거 
def clean_words(documents):
 
    //특수기호 제거
    for i, document in enumerate(documents):
        document = re.sub(r'[^ ㄱ-ㅣ가-힣]', '', document) //특수기호 제거, 정규 표현식
        documents[i] = document

    //Mecab 활용 형태소 분석
    mecab = konlpy.tag.Mecab()
    for i, document in enumerate(documents):
        
        clean_words = []
        for word in mecab.pos(document): #어간 추출
            if word[1] in ['NNG', 'VV', 'VA']: /명사, 동사, 형용사
                clean_words.append(word[0])
        document = ' '.join(clean_words)
        documents[i] = document

    //텍불용어 제거
    df = pd.read_csv('https://raw.githubusercontent.com/cranberryai/todak_todak_python/master/machine_learning_text/clean_korean_documents/korean_stopwords.txt', header=None)
    df[0] = df[0].apply(lambda x: x.strip())
    stopwords = df[0].to_numpy()

    for i, document in enumerate(documents):
        clean_words = [] 
        for word in nltk.tokenize.word_tokenize(document): 
            if word not in stopwords: //불용어 제거
                clean_words.append(word)
        documents[i] = ' '.join(clean_words)  

    return documents
  
#전처리하기
x_data = clean_words(x_data) 
#TF-IDF 적용
transformer = TfidfVectorizer()
transformer.fit(x_data)
#단어 카운트 가중치 적용
x_data = transformer.transform(x_data) 

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.1, random_state=777, stratify=y_data)

if not os.path.exists('models/kiosk_order_chat_bot_model'):
    os.makedirs('models/kiosk_order_chat_bot_model')

with open('models/kiosk_order_chat_bot_model/transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)
    
model = MultinomialNB(alpha=0.01) #smoothing
model.fit(x_train1, y_train1)    

with open('models/kiosk_order_chat_bot_model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
#의도 파악 - NB 배열 출력
def intent1(text):
    x_test = np.array([
        text
    ])
    #전처리
    x_test = clean_words(x_test) 
    #단어 카운트 가중치:tfidfvectorizer
    x_test = transformer.transform(x_test) 
    #예측하기
    y_predict = model.predict(x_test)
    #클래스 라벨
    label = labels[y_predict[0]]
    y_predict = model.predict_proba(x_test)
    #신뢰도
    confidence = y_predict[0][y_predict[0].argmax()]
    max=0;
    for i in range(22):
      if(y_predict[0][i]<confidence and max<y_predict[0][i]):
        max=y_predict[0][i]
        index=i
        label2=labels[index]
    return y_predict[0]    
  
#의도 파악 - NB 배열 출력
def NB_predict(text):
    x_test = np.array([
        text
    ])
    #전처리
    x_test = clean_words(x_test) 
    #단어 카운트 가중치:tfidfvectorizer
    x_test = transformer.transform(x_test) 
    #예측하기
    y_predict = model.predict(x_test)
    #클래스 라벨
    label = labels[y_predict[0]]
    y_predict = model.predict_proba(x_test)
    #신뢰도
    confidence = y_predict[0][y_predict[0].argmax()]
    max=0;
    for i in range(15):
      if(y_predict[0][i]<confidence and max<y_predict[0][i]):
        max=y_predict[0][i]
        index=i
        label2=labels[index]
    return ' {} {:.2f}% '.format(labels[y_predict[0].argmax()], y_predict[0][y_predict[0].argmax()]*100)  
  
  
//Bi-LSTM 

#bib=lstm 모델
#Modeling하기 - 양방향 LSTM 사용
#파라미터 설정
vocab_size = 2000
embedding_dim = 200
max_length = 15    # 위에서 그래프 확인 후 정함
padding_type='post'
model3 = tf.keras.models.Sequential([Embedding(vocab_size, embedding_dim, input_length =max_length),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64)),  #dropout
        Dense(22, activation='softmax')    # 결과값이 0~22 이므로 Dense(22)
    ])
    
model3.compile(loss= 'categorical_crossentropy', #여러개 정답 중 하나 맞추는 문제이므로 손실 함수는 categorical_crossentropy
              optimizer= 'adam',
              metrics = ['accuracy']) 
model3.summary()

history = model3.fit(train_x, train_y, epochs=50, batch_size=100, validation_split= 0.2) 

#bilstm 배열 출력
def intent2(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = np.array([text])
  test_lndex = tokenizer.texts_to_sequences(test_word)
  #print(test_word)
  #Check for unknown words
  if [] in test_lndex:
    test_Index = list(filter(None, test_lndex))
  max_length=15
  #test_x = pad_sequences(test_Index, padding='post', maxlen=max_length)
  #test_y = np.zeros((test_x.shape[0], 15))


  test_lndex = np.array(test_lndex)#.reshape(1, len(test_lndex))
 
  x = padding_doc(test_lndex, max_length)
  
  index=['주문 인텐트', '햄버거메뉴안내 인텐트', '음료메뉴안내 인텐트','주문확인 인텐트','취소 인텐트','상품취소 인텐트','상품 변경 인텐트','수량변경 인텐트','매장포장 인텐트','계산 인텐트','포인트 적립 인텐트','카드 인텐트','모바일 인텐트','음성인식/키오스크 사용방법 인텐트','매장 안내 인텐트','세트메뉴 안내/주문 인텐트','사이드 메뉴 안내/주문 인텐트','행사 인텐트','반응 인텐트','중간의도 인텐트','토핑 인텐트','가격 안내 인텐트','미매칭']
  #pred = model3.predict_proba(x)
  pred = model3.predict(x)
  #pred = model3.predict_proba(self, x, batch_size=32, verbose=0)
  max=0
  max1=0
  max2=0
  for i in range(22):
    x=pred[0][i]
    if max<x:
      max=x
      max_intent=i
 #print("1 %s 인텐트, 신뢰도: %s "%(index[max_intent],max*100))    
  #두번째로 높은것
  for j in range(22):
    x=pred[0][i]
    if j!=max_intent: 
      if max1<x :
        max1=x
        max_intent1=j
  #print("2 %s 인텐트, 신뢰도: %s "%(index[max_intent1],max1*100))   
 

  return pred[0]

#bilstm 
def lstm_predict(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = np.array([text])
  test_lndex = tokenizer.texts_to_sequences(test_word)
  #Check for unknown words
  if [] in test_lndex:
    test_Index = list(filter(None, test_lndex))
  max_length=15
  test_lndex = np.array(test_lndex)#.reshape(1, len(test_lndex))
  x = padding_doc(test_lndex, max_length)
  pred = model3.predict(x)
  return ' {} {:.2f}% '.format(labels[pred[0].argmax()], pred[0][pred[0].argmax()]*100)

//MultinomialNB+Bi-LSTM soft-voting

def prediction(text): 
  predict1=intent1(text)
  predict2=intent2(text)
  predictf = predict1 * 0.2+ predict2 * 0.8
  a=predictf[predictf.argmax()]
  if a>0.4:
    label = labels[predictf.argmax()]
    confidence =predictf[predictf.argmax()]
  else:
    label=labels[22]
    confidence=1-predictf[predictf.argmax()]  
 
  return ' {} {:.2f}% '.format(label, confidence * 100)

def predict(text):  #index만 리턴하는 함수
  predict1=intent1(text)
  predict2=intent2(text)
  predictf = predict1 * 0.2+ predict2 * 0.8
  a=predictf[predictf.argmax()]
  if a>0.4:
    label = labels[predictf.argmax()]
    confidence =predictf[predictf.argmax()]
    res=predictf.argmax()
  else:
    label=labels[22]
    res=22
    confidence=1-predictf[predictf.argmax()]  
 
  return res
