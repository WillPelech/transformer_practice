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





from torch import rand
from torch.nn import TransformerEncoder, TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
src = rand(10, 32, 512)
out = transformer_encoder(src) 
print(src)


