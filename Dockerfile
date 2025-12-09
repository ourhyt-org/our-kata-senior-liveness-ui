FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app/ ./app/

CMD ["app.main.handler"]