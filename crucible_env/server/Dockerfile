FROM openenv-base:latest
RUN pip install anthropic pydantic streamlit plotly pandas
COPY . /app/env
WORKDIR /app/env
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
