from flask import Flask, render_template, request, jsonify

from chat import get_response
from pygame import mixer
from gtts import gTTS
#import playsound
#from pydub import AudioSegment
#from pydub.playback import play
import os

app = Flask(__name__)
language = 'en'


def gtts_speech(response):
    myobj = gTTS(text=response, lang=language, slow=False, tld='com.sg')
    myobj.save("./welcome1.mp3")
    #os.system("mpg321 ./welcome.mp3")
    #os.system("./welcome1.mp3")
    #os.remove("./welcome1.mp3")
    mixer.init()
    mixer.music.load('./welcome.mp3')# load the audio file
    mixer.music.play()
    #playsound.playsound('welcome.mp3', True)
    #song = AudioSegment.from_mp3("./welcome1.mp3")
    #play(song)

@app.route("/")
def index_get():
    return render_template("base.html")


@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    gtts_speech(response)
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=False)
