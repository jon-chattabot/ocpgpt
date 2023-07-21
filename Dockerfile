# FROM registry.access.redhat.com/ubi9/python-311@sha256:381d9f3aa228ab406a1dff0dda4950c5f321a9312be37442a0770e62f651b7ba
FROM docker.io/python:3.11.4

COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .

EXPOSE 8000
CMD ["run.sh"]
