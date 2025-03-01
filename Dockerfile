FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

CMD ["python", "app.py"]
