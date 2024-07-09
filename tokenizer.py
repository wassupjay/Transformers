from transformers import BartModel, AutoTokenizer
import pandas as pd 

model_name="bert-base-uncased"
model=BartModel.from_pretrained (model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name) 

sentence="when life gives you lemons, don't make lemonade"
tokens=tokenizer.tokenize(sentence)
tokens
vocab=tokenizer.vocab
vocab_df=pd.DataFrame({"token": vocab.keys(),"token_id":vocab.values()})
vocab_df