FROM python:3.10-buster

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/code

CMD ["sh", "entrypoint.sh"]