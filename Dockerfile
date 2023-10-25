FROM registry.access.redhat.com/ubi9/python-311@sha256:381d9f3aa228ab406a1dff0dda4950c5f321a9312be37442a0770e62f651b7ba
USER root
RUN dnf install -y python3-devel

COPY requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .

RUN chown -R 1001 .
EXPOSE 8000

USER 1001
ENV OPENAI_API_KEY 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
ENV SYSTEM_TEMPLATE "Answer the following questions as best you can. You have access to the following tools:\n\ntoday_tool: Useful for figuring out the date. Input should be a search query.\nrelative_date_tool: Useful for when you need to figure out what day a relative date is.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [today_tool, relative_date_tool]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the answer\nAnswer: the answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"
# RUN python ingest.py
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
