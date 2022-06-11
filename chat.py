import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

# from gtts import gTTS
# import os

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

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

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
gpt_model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')

bot_name = "Jude"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
            
    else:
        #encoded_input = tokenizer(msg, return_tensors='pt')['input_ids']
        new_user_input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors='pt')
        step=0
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        chat_history_ids = gpt_model.generate(
               bot_input_ids, max_length=500,
               pad_token_id=tokenizer.eos_token_id,
               no_repeat_ngram_size=3,
               do_sample=True,
               top_k=100,
               top_p=0.7,
               temperature=0.8
            )

        #print("AI: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        #output = model(**encoded_input)
        return output   

    #return "I do not understand..."


# def gtts_speech(response):
#     myobj = gTTS(text=response, lang=language, slow=False, tld='com.sg')
#     myobj.save("welcome.mp3")
#     os.system("mpg321 welcome.mp3")


# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         gtts_speech(resp)
#         print(resp)
