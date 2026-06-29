FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --no-root

COPY app.py .
COPY model.keras .

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
