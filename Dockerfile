FROM python:3.14-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app:/home/user/app/src

WORKDIR $HOME/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user dashboard ./dashboard
COPY --chown=user src ./src

EXPOSE 7860

CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "7860"]
