FROM python:3.5

EXPOSE 8080
EXPOSE 5000

COPY . /app
RUN cd /app; pip3 install -r /app/requirements.txt

WORKDIR /app

CMD ["python", "src/webapp/webapp.py"]
