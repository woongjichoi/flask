# https://thecodex.me/blog/speech-recognition-with-python-and-flask
# 1. 사용자로부터 입력 오디오 파일을 받는 간단한 Flask 웹 어플리케이션 빌드하기 (21-05-18)
# 2. 오디오 파일 분석 및 텍스트 변환 (21-05-19)
# 3. 전사 + 마지막 터치 표시 (21-05-21)

# <개선점>
# 입력받은 오디오 파일 분석 → 마이크에서 입력받은 음성 분석
# 파이썬 음성 인식 라이브러리 SpeechRecognition(https://pypi.org/project/SpeechRecognition/) → Google Cloud STT API
# transcribe 시간 단축

from flask import Flask, render_template, request, redirect
import speech_recognition as sr # pip3 install --upgrade speechrecognition

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript=""
    if request.method=="POST":
        print("FORM DATA RECEIVED")

        # 요청의 POST 메소드에 "file"이 있는지 확인
        if "file" not in request.files:
            return redirect(request.url)

        file=request.files["file"] # 입력받은 오디오 파일
        # 완료되면 파일에 실제로 파일 이름 있는지 확인
        if file.filename=="":
            return redirect(request.url)

        # 파일이 성공적으로 전달되면 분석 가능한 형식으로 변환
        if file:
            recognizer=sr.Recognizer()
            audioFile=sr.AudioFile(file) # 입력받은 오디오 파일에 대해 SpeechRecognition의 AudioFile 인스턴스 반환
            with audioFile as source:
                data=recognizer.record(source)
            transcript=recognizer.recognize_google(data, key=None)
            # Google의 음성 인식 API 실행
            # 기본 인식기 .recognize_google은 대략 1분 미만의 오디오 텍스트 변환을 허용함
            # 더 큰 파일을 분석하려면 실제 API 키 지정 or Google API 키에서 유료 라이선스로 업그레이드
            # (https://pypi.org/project/SpeechRecognition/)
            print(transcript)

    return render_template('index.html', transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)