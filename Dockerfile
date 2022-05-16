FROM python:3.7.13

RUN apt -y update && apt -y upgrade
RUN apt -y install libsndfile1

RUN mkdir -p /app

COPY ./requirements.txt /app

RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY . /app

ENV PORT=8000
# ENV APP_ENV=production

EXPOSE 8000

# CMD [ "/bin/bash"]