import requests
from dotenv import load_dotenv
import os 
import pandas as pd


load_dotenv()

model_id = "sentence-transformers/all-MiniLM-L6-v2"


#### Setting up the embedding 
hf_token= os.getenv("hf_token")
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()




text = ["test to see if we can get the embdeddings to work"]
embedding_query = query(text)
embeddings =pd.DataFrame(embedding_query)
print(embeddings)

#TODO Reranker

#### positional encoding 
import 



