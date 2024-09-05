import requests
import memory
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()

key = os.getenv("HUGG_KEY")

API_URL = "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {key}"}

chat_mem = memory.Memory(1000)

def query(payload):
    response = requests.post(API_URL, headers= headers, json=payload)


def gen_response(user_input):
    chat_mem.add_to_memory(usr_input= user_input, is_user= True)

    context = chat_mem.get_memory()

    respose = query({
        "inputs": context,
        "parameters": {
            "max_new_tokens": 150,
            "return_full_text": False
        }
    })

    generated_text = respose["generated_text"]

    chat_mem.add_to_memory(generated_text, is_user=False)

    return generated_text