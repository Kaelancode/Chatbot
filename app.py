from flask import Flask, render_template, request, jsonify
import json
from chat import get_response
#from gtts import gTTS
import os

app = Flask(__name__)
language = 'en'


# def gtts_speech(response):
#     myobj = gTTS(text=response, lang=language, slow=False, tld='com.sg')
#     myobj.save("welcome.mp3")
#     os.system("mpg321 welcome.mp3")
global COUNTS
COUNTS = [12, 19, 3, 5, 2, 10]
jobs_ans = "There will be no placement of jobs but there will career development support as well as sharing of job opportunities with our industry partners.<br><br><a href='https://aisingapore.org/kb/will-i-get-a-job-after-the-programme/'>Regarding job placements</a>"
outcome_ans = "Candidates should be equipped and be proficient in all or most of the following skills:<br><br>Data engineering<br>Model development<br>Software engineering<br>MLOps<br><br><a href='https://aisingapore.org/kb/what-is-the-outcome-of-this-programme/' target='_blank'>Training outcome</a>"
dropout_ans = "As AIAP is a fully sponsored programme, if you drop out of the programme, you will have to reimburse AI Singapore for the monthly training allowance that has been paid out to you and the programme fees."
fulltime_ans = "AIAP is a FULL-TIME programme where you will undergo training at an intensive pace over 9/12 months at AI Singapore's office / any appropriate location. <br><br> Due to our sponsorship requirements, you are not allowed to take no-pay leave from your current full-time job to attend the programme. In addition, you should not be owner/founder of a company. Apprentices work on industry projects from companies and NDAs have to be signed. Hence during the programme period, apprentices should not be associated with any company."
candidates_ans = "We are looking for candidates who possess a keen interest to pursue a career in the area of machine learning and data science.<br><br><a href='https://aisingapore.org/kb/what-type-of-candidates-are-you-looking-for/' target='_blank'>Who we are seeking</a>"
salary_ans = "The monthly training allowance is between SGD 3500 to SGD 5500 (dependent on no. of years of working experience and qualification)."


@app.get("/")
def index_get():
    return render_template("base_chart.html", chart_data=json.dumps(COUNTS))


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    if message["answer"] == salary_ans:
        print('Match salary_ans')
        COUNTS[0] = COUNTS[0]+1

    elif message["answer"] == dropout_ans:
        print('Match dropout_ans')
        COUNTS[1] = COUNTS[1]+1

    elif message["answer"] == candidates_ans:
        print('Match candidates_ans')
        COUNTS[2] = COUNTS[2]+1

    elif message["answer"] == fulltime_ans:
        print('Match fulltime_ans')
        COUNTS[3] = COUNTS[3]+1

    elif message["answer"] == outcome_ans:
        print('Match outcome_ans')
        COUNTS[4] = COUNTS[4]+1

    elif message["answer"] == jobs_ans:
        print('Match jobs_ans')
        COUNTS[5] = COUNTS[5]+1

    # gtts_speech(response)
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=False)
