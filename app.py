from dotenv import load_dotenv
from os import environ, path
from re import sub
import re
from datetime import datetime
from dateutil import parser
import calendar
import openai
import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.llms import OpenAI, SelfHostedHuggingFaceLLM, LlamaCpp
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain import LLMMathChain
from chainlit.server import app
from fastapi import Request
from fastapi.responses import HTMLResponse
load_dotenv()

system_template = environ.get("SYSTEM_TEMPLATE", "You are a helpful bot. If you do not know the answer, just say that you do not know, do not try to make up an answer.")
embedding_model_name = environ.get("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')
embedding_type = environ.get("EMBEDDING_TYPE", 'openai')
show_sources = environ.get("SHOW_SOURCES", 'True').lower() in ('true', '1', 't')
retrieval_type = environ.get("RETRIEVAL_TYPE", "conversational")  # conversational/qa
verbose = environ.get("VERBOSE", 'True').lower() in ('true', '1', 't')
stream = environ.get("STREAM", 'True').lower() in ('true', '1', 't')

model_path = environ.get("MODEL_PATH", "")
model_id = environ.get("MODEL_ID", "gpt2")
openai.api_key = environ.get("OPENAI_API_KEY", "")
botname = environ.get("BOTNAME", "OCP-GPT")
temperature = float(environ.get("TEMPERATURE", 0.0))

# Date functions
def today_date() -> str:
    return datetime.now().strftime('%m/%d/%y')

# Get day of week for a date (or 'today')
def day_of_week(date):
    if date == 'today':
        return calendar.day_name[datetime.now().weekday()]
    else:
        date_pattern = re.compile(r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{2}$')
        if date_pattern.match(date):
            try:
                theDate = parser.parse(date)
                return calendar.day_name[theDate.weekday()]
            except:
                return 'invalid date, unable to parse'
        else:
            return 'invalid date format, please use format: mm/dd/yy'

# Helpers
def create_chain() -> (BaseConversationalRetrievalChain | BaseRetrievalQA):
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

    output_key = "result"
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key=output_key, return_messages=True)
    return_source_documents = show_sources

    # llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    # math_tool = Tool.from_function(
    #     func=llm_math_chain.run,
    #     name="Calculator",
    #     description="Useful for when you need to answer questions about math. This tool is only for calculating the day in which the user input wants to know. This tool is for math questions and nothing else. For example, the user asks what day will it be 2 days from now, that is when you use this tool"
    # )
    today_tool = Tool(
        name = "today's date",
        func = lambda string: today_date(),
        description="use to get today's date",
        )
    relative_date_tool = Tool(
        name = "day of the week",
        func = lambda string: day_of_week(string),
        description="use to get the day of the week, input is 'today' or any relative date like 'tomorrow' ",
        ) 
        
    zero_shot_agent = initialize_agent(
        tools = [today_tool,relative_date_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=4,
        stop=["\nObservation:"],
        handle_parsing_errors="Check you output and make sure it conforms! Do not output an action and a final answer at the same time."
        )
    
    zero_shot_agent.agent.llm_chain.prompt.template = '''
    Answer the following questions as best you can using our database. \
    You have access to the following tools:

    today's date: Use it to find the date of today. \
                    Always used as first tool
    day of the week: Use this to find the day of a week where the input is 'today' or any relative date like 'tomorrow'. \
            Use it only after you have tried using the today's date tool.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [today's date, day of the week]. \
            Always use today's date first.
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the answer
    Final Answer: the answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    '''   
    
    if retrieval_type == "conversational":
        conversation_template = """Combine the chat history and follow up question into a standalone question.
Chat History: ({chat_history})
Follow up question: ({question})"""
        condense_prompt = PromptTemplate.from_template(system_template + "\n" + conversation_template)

        # https://github.com/langchain-ai/langchain/issues/1800
        # https://stackoverflow.com/questions/76240871/how-do-i-add-memory-to-retrievalqa-from-chain-type-or-how-do-i-add-a-custom-pr
        return (ConversationalRetrievalChain.from_llm(
                llm,
                retriever,
                memory=memory,
                output_key=output_key,
                verbose=verbose,
                return_source_documents=return_source_documents,
                condense_question_prompt=condense_prompt), zero_shot_agent)
    else:
        messages = [
            SystemMessagePromptTemplate.from_template(system_template + "  Ignore any context like {context}."),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        chain_type_kwargs = {
                "prompt": ChatPromptTemplate.from_messages(messages),
                "memory": memory,
                "verbose": verbose,
                "output_key": output_key
                }
        return (RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=return_source_documents,
            verbose=verbose,
            output_key=output_key,
            chain_type_kwargs=chain_type_kwargs), zero_shot_agent)

def create_embedding_and_llm(embedding_type:str, model_path:str = "", model_id:str = "", embedding_model_name:str = ""):
    """
    Create embedding and llm
    """
    embedding = None
    llm = None

    match embedding_type:
        case "llama":
            llm = LlamaCpp(model_path=model_path, seed=0, n_ctx=2048, max_tokens=512, temperature=temperature, streaming=stream)
            embedding = LlamaCppEmbeddings(model_path=model_path)
        case "openai":
            llm = OpenAI(temperature=temperature, streaming=stream)
            embedding = OpenAIEmbeddings()
        case "huggingface":
            # gpu = runhouse.cluster(name="rh-a10x", instance_type="A100:1")
            # llm = SelfHostedHuggingFaceLLM(model_id=model_id, hardware=gpu, model_reqs=["pip:./", "transformers", "torch"])
            llm = OpenAI(temperature=temperature)
            embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return (llm, embedding)

def process_response(res:dict) -> list:
    """ Format response """
    elements:list = []
    if show_sources and res.get("source_documents", None) is not None:
        for source in res["source_documents"]:
            src_str:str = source.metadata.get("source", "/").rsplit('/', 1)[-1]
            final_str:str = f"Page {str(source.page_content)}"
            elements.append(cl.Text(content=final_str, name=src_str, display="inline"))
    if verbose:
        print("process_response")
        print(elements)
    return elements

# App Hooks
@cl.on_chat_start
async def main() -> None:
    ''' Startup '''
    openai.api_key = environ["OPENAI_API_KEY"]
    await cl.Avatar(
        name=botname,
        path="public/chattabot-logo.png"
    ).send()
    await cl.Message(
        content=f"Ask me anything about Rosebar.", author=botname
    ).send()

    (chain, agent) = create_chain()
    cl.user_session.set("llm_chain", chain)
    cl.user_session.set("zero_shot_agent", agent)

@cl.on_message
async def on_message(message:str) -> None:
    llm_chain:(BaseConversationalRetrievalChain | BaseRetrievalQA) = cl.user_session.get("llm_chain")
    agent = cl.user_session.get("zero_shot_agent")
    result = agent.run(message)
    if verbose:
        print(result)

    res = await llm_chain.acall(message + " " + result, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=stream)])
    if verbose:
        print(res)
    content = res["result"]
    content = sub("^System: ", "", sub("^\\??\n\n", "", content))
    if verbose:
        print("main")
        print(f"result: {res['result']}")

    await cl.Message(content=content, elements=process_response(res), author=botname).send()

# Custom Endpoints
@app.get("/botname")
def get_botname(request:Request) -> HTMLResponse:
    if verbose:
        print(f"calling botname: {botname}")
    return HTMLResponse(botname)
