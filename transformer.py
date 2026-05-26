from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os 
import pandas as pd

DENOM_BASE = 10000

load_dotenv()

model_id = "sentence-transformers/all-MiniLM-L6-v2"

#### Setting up the embedding 
hf_token = os.getenv("hf_token")
client = InferenceClient(api_key=hf_token)

def query(texts):
    return client.feature_extraction(texts, model=model_id)




text = ["test to see if we can get the embdeddings to work"]
embedding_query = query(text)
embeddings = pd.DataFrame(embedding_query)
print(embeddings)

#TODO Reranker

#### positional encoding 
## so this equation is used p(k,j)
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from math import sin, cos 
#positional encoder

#Module Inherited type

class PositionalEncoder(Module):
    def __init__(self, dimension = 512):
        super().__init__()
        self.dimension = dimension 
        
    def forward(self,text:list[str])->Tensor:
        res = []
        for index,_ in enumerate(text):
            res.append(self.get_positional_encoding(index))
        return Tensor(res) 
    
    #the idea is that we provide the whole row for a specific entry 
    def get_positional_encoding(self, k) -> list[float]:
        res = []
        for j in range(self.dimension):
            if j % 2 == 0:
                res.append(self.even_index(k, j))
            else:
                res.append(self.odd_index(k, j))
        return res

    def even_index(self, k, j):
        i = j // 2
        return sin(k / (DENOM_BASE ** (2 * i / self.dimension)))

    def odd_index(self, k, j):
        i = j // 2
        return cos(k / (DENOM_BASE ** (2 * i / self.dimension)))




pos = PositionalEncoder()
result_tensor = pos(text[0].split())



print("============Results================")
print(f"Embedding Query = {embedding_query}")
print(f"PositonalEncoding = {result_tensor}")

