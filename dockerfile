FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
COPY .env /app
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["python", "-m", "flask", "run"]