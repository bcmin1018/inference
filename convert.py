import os
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", return_dict=False)

text = "Sample Input for Tracing"
encoded_input = tokenizer(text, return_tensors='pt')
folder_save = "twitter-roberta-base-sentiment"
os.makedirs(folder_save, exist_ok=True)

cpu_traced = torch.jit.trace(model, (encoded_input["input_ids"], encoded_input["attention_mask"]))
# print(cpu_traced.graph)
# print(cpu_traced.code)

torch.jit.save(cpu_traced, os.path.join(folder_save, "model.pt"))
# tokenizer.save_pretrained(folder_save)
