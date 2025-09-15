import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import re
import html

path = "data/data.csv"
df = pd.read_csv(path)

print("data success")

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

def clean_df(text):
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'<.*?>', '', text)

    text = html.unescape(text)

    text = re.sub(r'https\S+|https\S+', 'url', text)

    return text.strip()

df['cleaned_text'] = df['Sentence'].apply(clean_df)
device = nn.device("cuda" if nn.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
model.eval()
