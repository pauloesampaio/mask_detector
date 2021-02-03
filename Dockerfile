FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app/app/
COPY ./app/requirements.txt .
RUN apt-get update && apt-get install -y \ 
    build-essential \ 
    ffmpeg \
    libsm6 \
    libxext6
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install orjson==3.4.7
COPY ./app/ /app/app/
