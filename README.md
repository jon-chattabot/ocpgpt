# Chattabot GPT

Chattabot GPT Solution

## Prerequisite

1. Python 3.11

## Setup

1. Clone this repo

2. Create virtual environment

    ```shell
    python -m venv venv
    source ./venv/bin/activate
    ```

3. Install dependencies

    ```shell
    pip install -r requirements.txt
    ```

4. Create an `.env` file that looks like

    ```ini
    OPEN_API_KEY=YOUR_KEY_HERE
    VERBOSE=false
    SHOW_SOURCES=false
    RETRIEVAL_TYPE=conversational
    SYSTEM_TEMPLATE="You are a helpful bot. If you do not know the answer, just say that you do not know, do not try to make up an answer."
    TEMPERATURE=0.0
    ```

### Ingesting

1. Move all documents that you want to index into the `docs` directory

2. Run the ingester

   ```shell
   python ./ingest.py
   ```

### Running the App

1. Start the app using chainlit

    ```shell
    chainlit run app.py
    ```
