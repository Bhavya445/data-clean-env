FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2

EXPOSE 7860

CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS