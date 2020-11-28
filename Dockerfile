FROM alpine:3.12
WORKDIR /home/python

RUN apk add py3-numpy py3-flask

COPY src src
ENV FLASK_ENV=production
ENV FLASK_APP=src/web.py
CMD ["python3", "src/web.py"]
