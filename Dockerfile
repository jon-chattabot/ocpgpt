FROM registry.access.redhat.com/ubi9/python-311@sha256:381d9f3aa228ab406a1dff0dda4950c5f321a9312be37442a0770e62f651b7ba
USER root
RUN dnf install -y python3-devel

COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .

RUN chown -R 1001 .
EXPOSE 8000

USER 1001
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
