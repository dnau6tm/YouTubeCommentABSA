import time
import streamlit as st
import numpy as np
import pickle
import pandas as pd

from model import SentimentAnalysisModelViSo

from transformers import AutoModel, AutoTokenizer

import torch
from torch import nn



def main():
     pkl_file = open(f'./mlb.pkl', 'rb')
     mlb = pickle.load(pkl_file)
     pkl_file.close()
     model, tokenizer = load_model()
     st.title("Aspect Based Sentiment Analysis")
     # Load model
     text_input = st.text_input("Nhập đầu vào:")
     if st.button("Predict"):
          encoded_data = tokenizer([text_input], padding=True, truncation=True, max_length=256, add_special_tokens = True, return_tensors="pt")
          outputs = model(encoded_data['input_ids'], encoded_data['attention_mask'])
          prob = torch.sigmoid(outputs).cpu().detach().numpy()
          st.text(prob)

          for k in range(len(prob)):
               if max(prob[k]) < 0.5:
                    prob[k] = np.where(prob[k]>max(prob[k])-1e-4, 1, 0)

          mapped_array = np.where(prob > 0.5, 1, 0)
          original_labels = mlb.inverse_transform(mapped_array)[0]
          st.header('Result')
          st.text(original_labels)


@st.cache_resource  # hash_func
def load_model():
     print("Loading model ...")
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     list_emoji = pd.read_csv('./newtokens.txt', header=None)


     # Load pre-trained PhoBERT model and tokenizer
     viso_model= AutoModel.from_pretrained('uitnlp/visobert')
     viso_tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

     viso_tokenizer.add_tokens(list(list_emoji[0]))
     viso_model.resize_token_embeddings(len(viso_tokenizer))


     absa_model = SentimentAnalysisModelViSo(viso_model, 12).to(device)
     absa_model.load_state_dict(torch.load('./absa_model_visobert.pth', map_location=torch.device('cpu')))
     return absa_model, viso_tokenizer


if __name__ == "__main__":
    main()