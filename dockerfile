FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
VOLUME /app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
ADD . /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]