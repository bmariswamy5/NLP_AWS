import argparse
import os
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('punkt')
# ----------------------------------------------------------------------------------------------------------------------
with open('sample.txt','r') as f:
    text=f.read()
sentences=sent_tokenize(text)
sentence_data   = {'Sentence': sentences, 'Word Count': [len(word_tokenize(sentence)) for sentence in sentences]}
dataframe=pd.DataFrame(sentence_data)

output= "Text Feature"
os.makedirs(output,exist_ok=True)

res=os.path.join(output,'sent.csv')
dataframe.to_csv(res,index=False)
print(f"CSV file saved at;{res}")