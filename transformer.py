from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os 
import pandas as pd


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
        for word,index in enumerate(text):
            res.append(get_positional_encoding())
            
        

        return Tensor([]) 
    
    #the idea is that we provide the whole row for a specific entry 
    def get_positional_encoding(self, index)->list[int]:
        dim = self.dimension
        for i in range(dim):
            if i%2==0:
                self.odd_index()
            elif i%2==1:
                self.even_index()
            

        return [1]
        
    def odd_index(self):
        pass

    def even_index(self):
        pass




print(src)


