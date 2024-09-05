import requests
import memory
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()

key = os.getenv("HUGG_KEY")

API_URL = ""
headers = {"Authorization": f"Bearer {key}"}

mem = memory.Memory(1000)
mem.add_to_memory("hello", is_user= False)

print(mem.get_memory())