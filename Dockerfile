FROM python:3.10.6-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /usr/local/share/nltk_data
ENV NLTK_DATA=/usr/local/share/nltk_data

RUN python -m nltk.downloader stopwords

COPY . .
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
