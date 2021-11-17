from __future__ import division
# https://thecodex.me/blog/speech-recognition-with-python-and-flask
# https://github.com/Uberi/speech_recognition/blob/master/examples/microphone_recognition.py
# https://dev.to/siddheshshankar/build-a-text-to-speech-service-with-python-flask-framework-3966

# 파이썬 음성 인식 라이브러리 SpeechRecognition(https://pypi.org/project/SpeechRecognition/) → Google Cloud STT API
# https://webnautes.tistory.com/1247

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

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        global transcript
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            #return render_template('index.html', transcript=transcript)
            break

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            #if re.search(r"\b(끝|종료)\b", transcript, re.I):
                #print("Exiting..")
                #break

            num_chars_printed = 0

@app.route("/", methods=["GET", "POST"])
def index():
    language_code = "ko-KR"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )  # single_utterance=True를 삭제하면 말하는 족족 계속 출력됨

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)

    return render_template('index.html', transcript=transcript)

'''
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
'''

# TTS
@app.route("/tts", methods=["GET", "POST"])
@cross_origin()
def homepage():
    if request.method=='POST':
        # text="안녕" 이라고 하면 http://192.168.0.11:5000/tts 에서 뭘 입력하든 "안녕"을 읽어줌
        text=request.form['speech']
        gender=request.form['voices']
        text_to_speech(text, gender)
        print(text)
        return render_template('frontend.html')
    else:
        return render_template('frontend.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)
