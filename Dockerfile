FROM public.ecr.aws/lambda/python:3.10

RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app/ ./app/

CMD ["app.main.handler"]