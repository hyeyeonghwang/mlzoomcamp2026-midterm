
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY predict.py .
COPY ./data/caffeine_crash_and_anxiety_telemetry.csv .

RUN python train.py

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
