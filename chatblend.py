import random
import json
#from gtts import gTTS
import os
import torch

from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

language = 'en'
FILE = "data.pth"
data = torch.load(FILE, map_location='cpu')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

mname = 'facebook/blenderbot_small-90M'
model_blender = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

bot_name = "Hey JUDE"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    print('X', X[0])
    output = model(X)
    print('output', output)
    _, predicted = torch.max(output, dim=1)
    print('predicted', predicted.item())
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    print("PROBS", probs)
    prob = probs[0][predicted.item()]
    print("Prob", prob)
    if prob.item() > 0.97:
        #print
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        inputs = tokenizer(msg, return_tensors='pt')
        reply_ids = model_blender.generate(**inputs)
        output = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return output

# def gtts_speech(response):
#     myobj = gTTS(text=response, lang=language, slow=False, tld='com.sg')
#     myobj.save("welcome.mp3")
#     os.system("mpg321 welcome.mp3")


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        #gtts_speech(resp)
        print(resp)
