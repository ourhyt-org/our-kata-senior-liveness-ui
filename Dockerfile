FROM public.ecr.aws/lambda/python:3.10

RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app/main.py ./main.py
COPY app/engine ./engine
COPY app/utils ./utils

CMD ["main.handler"]