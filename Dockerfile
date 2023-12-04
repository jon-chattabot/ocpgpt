FROM registry.access.redhat.com/ubi9/python-311@sha256:944fe5d61deb208b58dbb222bbd9013231511f15ad67b193f2717ed7da8ef97b
USER root
RUN dnf install -y python3-devel
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .

RUN chown -R 1001 .
EXPOSE 8000

USER 1001
# RUN python ingest.py
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
