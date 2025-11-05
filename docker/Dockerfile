FROM python:3.9.6
WORKDIR /app

RUN pwd

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src ./src/
COPY ./configs ./configs/

EXPOSE 10000
