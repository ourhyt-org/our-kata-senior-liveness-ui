# Imagen base de AWS Lambda para Python 3.10 (ARM64)
FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

# Instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Handler para AWS Lambda
CMD ["app.lambda_function.handler"]
