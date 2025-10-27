FROM agrigorev/zoomcamp-model:2025

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
RUN pip install gunicorn

COPY ["predict_test.py", "pipeline_v1.bin", "./"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict_test:app"]