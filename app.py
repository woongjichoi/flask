# https://thecodex.me/blog/speech-recognition-with-python-and-flask
# 1. 사용자로부터 입력 오디오 파일을 받는 간단한 Flask 웹 어플리케이션 빌드하기 (21-05-18)
# 2. 오디오 파일 분석 및 텍스트 변환 (21-05-19)
# 3. 전사 + 마지막 터치 표시 (21-05-21)
# https://github.com/Uberi/speech_recognition/blob/master/examples/microphone_recognition.py
# 기존의 입력받은 오디오 파일을 분석하는 방식에서 마이크에서 입력받은 음성을 분석하는 방식으로 변환 (21-05-26)
# HTML, CSS 수정 (21-05-28)
# https://dev.to/siddheshshankar/build-a-text-to-speech-service-with-python-flask-framework-3966
# pyttsx3 이용하여 TTS 구현 (21-05-28, 21-05-31)

# <개선점>
# (해결) 1. 입력받은 오디오 파일 분석 → 마이크에서 입력받은 음성 분석
# (해결) 1-1. HTML, CSS 수정
# (해결) 2. pyttsx3 한국어 지원
# (해결) 2. 시끄러운 상황(ex. 노래 틀어놓기)에서도 음성 인식되는지 검증 ★
# 3. 파이썬 음성 인식 라이브러리 SpeechRecognition(https://pypi.org/project/SpeechRecognition/) → Google Cloud STT API
# 3. 파이썬 텍스트 음성 변환 라이브러리 pyttsx3 → Google Cloud TTS API
# 4. transcribe 시간 단축 (음성 파일 길이 때문일 수도 있음!)

from flask_cors import cross_origin
from flask import Flask, render_template, request, redirect
# NOTE: this example requires PyAudio because it uses the Microphone class
#   - 윈도우는 PyAudio 설치 시 http://dslab.sangji.ac.kr/?p=2550과 같은 방법으로 해결해야 하는데 캡처와 같은 오류 발생하여 경로에 접근
#   - what is the current encoding of the file? ansi
#   - 주석보다 최상단에 #coding=<utf-8> 입력 후 다시 pipwin install pyaudio 실행하여 오류 해결
import speech_recognition as sr # STT
import pyttsx3 # TTS

app = Flask(__name__)

def text_to_speech(text, gender):
    voice_dict = {'Male': 0, 'Female': 1}
    code = voice_dict[gender]

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # 말하기 속도
    engine.setProperty('volume', 1)  # 볼륨 (min=0, MAX=1)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[code].id)

    engine.say(text)
    engine.runAndWait()


# STT
@app.route("/", methods=["GET", "POST"])
def index():
    transcript=""

    # 마이크에서 오디오 얻기
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio=r.listen(source)

    # Google Speech Recognition을 이용해 음성 인식하기
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    try:
        transcript=r.recognize_google(audio, language="ko-KR")
        print("Google Speech Recognition thinks you said "+transcript)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return render_template('index.html', transcript=transcript)

# TTS
@app.route("/tts", methods=["GET", "POST"])
@cross_origin()
def homepage():
    if request.method=='POST':
        text=request.form['speech']
        gender=request.form['voices']
        text_to_speech(text, gender)
        print(text)
        return render_template('frontend.html')
    else:
        return render_template('frontend.html')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)