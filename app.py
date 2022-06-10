from flask import Flask, render_template, request, jsonify

from chat import get_response
from pygame import mixer
from gtts import gTTS
import os

app = Flask(__name__)
language = 'en'


def gtts_speech(response):
    myobj = gTTS(text=response, lang=language, slow=True, tld='com.sg')
    myobj.save("welcome.mp3")
    os.system("mpg123 welcome.mp3")
    #os.system("start welcome.mp3")
    #mixer.init()
    #mixer.music.load('welcome.mp3')# load the audio file
    #mixer.music.play()



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


# if __name__ == "__main__":
#     app.run(debug=False)
