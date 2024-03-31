# inference
사용 데이터 다운로드 경로: https://github.com/smilegate-ai/korean_unsmile_dataset?ref=breezymind.com

학습 도커 실행
$ docker build -t pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel-ap -f train.Dockerfile .
$ docker run -it -v $(pwd):/workspace --gpus=1 pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel-ap /bin/bash
