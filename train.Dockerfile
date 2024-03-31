FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN pip install --upgrade pip
RUN pip install transformers==4.18.0 --no-cache-dir