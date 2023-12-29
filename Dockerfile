FROM python:3.11.6

WORKDIR /workspace

ADD requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

ADD embeddings.py /workspace/

ADD app.py /workspace/

COPY clip_chroma /workspace/clip_chroma

COPY dataset /workspace/dataset

CMD ["gradio", "/workspace/app.py"]

