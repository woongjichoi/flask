2021-2 capstone design preject 그로쓰4팀 키오조

# 프로젝트 개요
 
무인 터치스크린 키오스크 미적응자들을 위한 음성인식 AI 점원 API 서비스 

## 주요 기능

- 음성인식 및 음성합성을 통한 대화
- FAQ 질의응답

## 개발 환경 및 기술
- Google Colab
- 음성인식(Speech-To-Text): Google cloud Speech-To-Text
- 음성합성(Text-To-Speech): Pyttsx3
- 의도 파악 모델: Bi-LSTM,Mutinomial Naive Bayes 
- 개체명 인식 모델: Bi-LSTM+CRF
- 답변 생성 모델: Transformer
- 서버: Flask 
## API 시스템 구조도

![systemarchitecture](https://user-images.githubusercontent.com/61787171/142223280-59ecdffa-c486-4239-a324-4163cce4acb3.PNG)



## 파일 디렉토리 구조

```bash
├── README.md
├── STT,TTS
│   ├── .idea
│   │    ├── inspectionProfiles
│   │    │   └── profiles_settings.xml
│   │    ├── .gitignore
│   │    ├── flaskrestful.iml
│   │    ├── misc.xml
│   │    ├── modules.xml
│   │    └── vcs.xml
│   ├── static
│   │   └── styles
│   │       └── index.css
│   │
│   ├── templates
│   │   ├── frontend.html
│   │   └── index.html
│   ├── app.py
│   ├── stttts.ipynb
│   └── transcribe_streaming_mic.py
├── chatbot
│   ├── answer
│   │   └── Chatbot Transformer model.py
│   ├── intent
│   │   ├─ Intent_bilstm.csv
│   │   ├─ Mutinomial_tfidf.py
│   │   └── lstmNBsfvoting.py
│   ├─ ner
│   │   └─ Chatbot NER.py
│   └── scenario.py
└── dataset
    ├── dataset.py
    └── nerexcel.py

``` 

## Kiosk Demo
   
####  [안드로이드 ver.](https://github.com/woongjichoi/chatbotdemo)
####  [윈도우 ver.](https://github.com/sonoasy/Kiosk_window/tree/main)
