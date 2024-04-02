from scipy.special import softmax
from transformers import AutoTokenizer
import tritonclient.http as httpclient

import numpy as np

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

input_name = ['input__0', 'input__1']
output_name = 'output__0'

def run_inference(sentence):

    triton_client = httpclient.InferenceServerClient(url="localhost:8000")

    desired_dims = (1, 256)
    inputs = tokenizer(sentence, max_length=256, padding='max_length', return_tensors='pt')
    input_ids = inputs['input_ids'].numpy()
    input_ids = np.reshape(input_ids, desired_dims).astype(np.int32)
    mask = inputs['attention_mask'].numpy()
    mask = np.reshape(mask, desired_dims).astype(np.int32)

    input0 = httpclient.InferInput(input_name[0], desired_dims, 'INT32')
    input0.set_data_from_numpy(input_ids, binary_data=False)
    input1 = httpclient.InferInput(input_name[1], desired_dims, 'INT32')
    input1.set_data_from_numpy(mask, binary_data=False)
    output = httpclient.InferRequestedOutput(output_name)

    response = triton_client.infer(model_name="roberta", inputs=[input0, input1], outputs=[output])
    logits = response.as_numpy('output__0')
    print(logits)
    scores = softmax(logits, axis=1)
    max_index = np.argmax(scores)

    return max_index


while True:
    sentence = input("> ").strip()
    result = run_inference(sentence)
    print(result)
