FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./backend/config.json /code/config.json


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./backend /code/app

CMD ["fastapi", "run", "app/api.py", "--port", "8088"]