from dotenv import load_dotenv
from os import environ, path
import openai
import chainlit as cl
from chainlit import Message, on_chat_start
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.llms import OpenAI, SelfHostedHuggingFaceLLM, LlamaCpp
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
load_dotenv()

SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


embedding_model_name = environ.get("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')
embedding_type = environ.get("EMBEDDING_TYPE", 'openai')
model_path = environ.get("MODEL_PATH", "")
model_id = environ.get("MODEL_ID", "gpt2")
openai.api_key = environ.get("OPENAI_API_KEY", "")

# Helpers
def create_embedding_and_llm(embedding_type:str, model_path:str = "", model_id:str = "", embedding_model_name:str = ""):
    """
    Create embedding and llm
    """
    temperature = 0.0
    embedding = None
    llm = None
    match embedding_type:
        case "llama":
            llm = LlamaCpp(model_path=model_path, seed=0, n_ctx=2048, max_tokens=512, temperature=0.0, streaming=True)
            embedding = LlamaCppEmbeddings(model_path=model_path)
        case "openai":
            llm = OpenAI(temperature=temperature)
            embedding = OpenAIEmbeddings()
        case "huggingface":
            # gpu = runhouse.cluster(name="rh-a10x", instance_type="A100:1")
            # llm = SelfHostedHuggingFaceLLM(model_id=model_id, hardware=gpu, model_reqs=["pip:./", "transformers", "torch"])
            llm = OpenAI(temperature=temperature)
            embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return (llm, embedding)

@on_chat_start
async def main():
    ''' Startup '''
    openai.api_key = environ["OPENAI_API_KEY"]
    await cl.Avatar(
        name="OCP GPT",
        url="https://cloud.redhat.com/hubfs/images/logos/osh/Logo-Red_Hat-OpenShift-A-Reverse-RGB.svg",
    ).send()
    await Message(
        content=f"Ask me anything about OpenShift.", author="OCP-GPT"
    ).send()


@cl.langchain_factory(use_async=True)
def load_model() -> BaseRetrievalQA:
    """ Load model to ask questions of it """
    (llm, embeddings) = create_embedding_and_llm(
            embedding_type=embedding_type,
            model_path=model_path,
            model_id=model_id,
            embedding_model_name=embedding_model_name)

    root_dir = path.dirname(path.realpath(__file__))
    db_dir = f"{root_dir}/db"

    db = FAISS.load_local(db_dir, embeddings)
    retriever = db.as_retriever()

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


@cl.langchain_postprocess
async def process_response(res:dict) -> None:
    """ Format response """
    answer = res["result"]
    sources:list[Document] = res["source_documents"]
    elements:list = []
    for source in sources:
        src_str:str = source.metadata.get("source", "/").rsplit('/', 1)[-1]
        final_str:str = f"Page {str(source.page_content)}"
        elements.append(cl.Text(content=final_str, name=src_str, display="inline"))

    await cl.Message(content=answer, elements=elements, author="OCP-GPT").send()
