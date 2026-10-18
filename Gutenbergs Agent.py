# 向量存储
import os
import json
import time
import random
import threading
import subprocess

from pathlib import Path
from operator import add
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()

## 嵌入式数据库
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma


## Agent dependencies
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage , AIMessage
from langgraph.graph import add_messages,  StateGraph, START, END
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

## NER
import spacy

## RAG dependencies:
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

NEW_LINE = "\n"
GROQ_LLM_INFERENCE = "llama-3.2-3b-preview"
OLLAMA_LLM_INFERENCE = "llama3.2"
OLLAMA_HOST = 'http://0.0.0.0:11434'
LLM_EMBBEDDINGS = "nomic-embed-text"

SPACEY_MODEL = "en_core_web_sm"

LOCAL_ARCHIVE_PATH = "./"
LOCAL_DEFAULT_BOOK = "3300-0.txt"
WEB_DEFAULT_BOOK = "https://www.gutenberg.org/files/3300/3300-0.txt"

CHROMA_STORE = "wealth_of_nations"

INDEX_NAME = "wealthofnations"

IS_LOCAL_ENVIRONMENT = True

def ollama():
    os.environ['OLLAMA_HOST'] = OLLAMA_HOST
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "pull", LLM_EMBBEDDINGS])
    subprocess.Popen(["ollama", "pull", OLLAMA_LLM_INFERENCE])
    subprocess.Popen(["ollama", "serve"])

ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()

## 程序逻辑

if not IS_LOCAL_ENVIRONMENT:
  os.environ["GROQ_API_KEY"] = os.environ.get('GROQ_API_KEY')
  os.environ["PINECONE_API_KEY"] = os.environ.get('PINECONE_API_KEY')

def build_llm():
  if IS_LOCAL_ENVIRONMENT:
    llm = ChatOllama(model=OLLAMA_LLM_INFERENCE)
  else:
    llm = ChatGroq(model=GROQ_LLM_INFERENCE)
  return llm

llm = build_llm()
nlp = spacy.load(SPACEY_MODEL)

## vectorstore
def build_vector_store():
  if not IS_LOCAL_ENVIRONMENT:
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
  else:
    return Chroma(
        embedding_function=embeddings,
        persist_directory=f"{LOCAL_ARCHIVE_PATH}{CHROMA_STORE}"
      )

embeddings = OllamaEmbeddings(model=LLM_EMBBEDDINGS)
embeddings.base_url = os.getenv("OLLAMA_HOST")

vectorstore = build_vector_store()


## Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
def embbed_paragraphs(db):
    sample_docs = DirectoryLoader(
            "./",
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        ).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = text_splitter.split_documents(sample_docs)

    if not IS_LOCAL_ENVIRONMENT:
      # 使用Pinecone
      db.from_documents(documents, embedding=embeddings, index_name="wealthofnations")
    else:
      # Use Local Chroma
      db.from_documents(documents, embedding=embeddings)

    return documents

## State
class User(BaseModel):
  name: str
  gender: str
  age: int
  likes: list[str]
  dislikes: list[str]

class Host(User):
  sys_msg: str = None
  personality: list[str]

# 添加讨论
def add_discussions(discussions, new_messages):
  if not new_messages:
    return []
  return discussions + new_messages

# 讨论状态
class DiscussionState(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages] = []
  topic: str
  users: list[User]
  host: Host
  curr_discussion: Annotated[list[AnyMessage], add_discussions] = []
  discussion_summaries: Annotated[list[str], add] = []
  setEndDiscussion: bool = False
  random_anchor: str
  book_file_path: str = LOCAL_DEFAULT_BOOK

"""
节点函数：
ingest_source_material：下载并存储原始资料

preprocess_source_material：执行命名实体识别与嵌入操作

prepare_topic：从处理过的资料中挑选讨论主题
"""
## LangGraph States
def introduce(state):
  host = state['host']
  topic = state['topic']
  users = state['users']

  # 系统提示词，将AI设定为一个会议的主持人，第一个主题进行讨论，参会的人员及其信息和会议的主题，以此单程优化提示词，让用户进行深度思考
  sys_msg = f"""
  You are a meeting host leading a meeting.

  You will be leading a group of {len(users)} members.

  Your job is to ask a quesetion about the to the users to discuss.

  The following is your information, do not break your character.
  name: {host.name}
  gender: {host.gender}
  age: {host.age}
  personality: {host.personality}


  The following are the information of the club users:
  {NEW_LINE.join([f"{user.name} ({user.gender}, {user.age}" for user in users])}

  The topic that will be discussed is {topic}
  """

  if state.get('guide'):
    sys_msg += f"\n Follow this guide: \n{state.get('guide')}"

  host.sys_msg = sys_msg

  # 旁白消息：成员介绍
  narrator_msg = f"""
  Club members just arrived!
  Before we dive in, please greet the members, introduce yourself, and give brief introductions of the members.
  """
  response = llm.invoke([ # 将旁白提示词输入，有AI模型完成成员的介绍
    SystemMessage(content=sys_msg),
    HumanMessage(content=narrator_msg, name="narrator")
  ])
  response.name = host.name
  response.content = f"{host.name}: {response.content}"
  response.pretty_print()
  return {
    'message': [response],
    'host': host,
  }

def ingest_source_material(state):
  """
  从Gutenberg项目中下载原始资料，为后续的讨论做准备
  """
  import requests
  response = requests.get(WEB_DEFAULT_BOOK)

  source_file_path = Path(f"{LOCAL_ARCHIVE_PATH}{LOCAL_DEFAULT_BOOK}")
  with open(source_file_path, "w") as source_file:
    source_file.write(response.text)

  return state

def extract_entities(file_path):
    # 读取文本文件
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 使用spaCy来处理文本
    nlp.max_length = len(text)
    doc = nlp(text)

    # 提取文本中的实体并按照类型对其进行整理
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = set()
        entities[ent.label_].add(ent.text)

    for key, value in entities.items():
        entities[key] = list(value)

    return entities

# 预处理元数据
def preprocess_source_material(state):
  """
  Performs NER against source material
  对原始文本进行词性标注处理
  """
  embbed_paragraphs(vectorstore) # 从向量库中加载源数据

  entities_file_path = Path(f"{LOCAL_ARCHIVE_PATH}entities_{LOCAL_DEFAULT_BOOK}")

  found_entities = extract_entities(f"{LOCAL_ARCHIVE_PATH}{LOCAL_DEFAULT_BOOK}") # 加载文件中存在的实体数据
  json_entities = json.dumps(found_entities)  # 转为JSON

  with open(entities_file_path, "w") as output:
      output.write(json_entities)

  return state

def prepare_topic(state):
  entities_file_path = Path(f"{LOCAL_ARCHIVE_PATH}entities_{LOCAL_DEFAULT_BOOK}")

  with open(entities_file_path, "r") as entities_file:
    entities = json.loads(entities_file.read())

  persons = entities.get("PERSON")  # 从所有实体中提取出"参会人"
  size = len(persons)

  state['random_anchor'] = persons[random.randint(1, size)]
  return state

"""
条件边：
should_download_source_material：资料是否齐全
should_prepare_source_material：实体是否生成完毕
route_discussion：讨论后，用户是否想结束本轮讨论
"""
# 下载原材料 - 这是图中的边，依据条件，绕过模型跳转到不同处理节点
def should_download_source_material(state):
  source_file_path = Path(f"{LOCAL_ARCHIVE_PATH}{LOCAL_DEFAULT_BOOK}")

  if(not source_file_path.exists() or not source_file_path.is_file()):
    return "web_download"

  entities_file_path = Path(f"{LOCAL_ARCHIVE_PATH}entities_{LOCAL_DEFAULT_BOOK}")

  if(not entities_file_path.exists() or not entities_file_path.is_file()):
    return "embeddings_ner"

  return "prepare_topic"

# 预处理原材料 - 条件边
def should_prepare_source_material(state):
  entities_file_path = Path(f"{LOCAL_ARCHIVE_PATH}entities_{LOCAL_DEFAULT_BOOK}")

  if(not entities_file_path.exists() or not entities_file_path.is_file()):
    return "embeddings_ner"

  return "prepare_topic"

def ask_question(state):
  # 1. 从检索器中找到任何有趣的主题

  retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
  )

  if not IS_LOCAL_ENVIRONMENT:
    # Use Pinecone
    results = retriever.invoke(state['random_anchor'], filter={"source": "wealthofnations"})
  else:
    # Use Local Chroma
    results = retriever.invoke(state['random_anchor'])


  # referenced_document = list(results)[0].page_content
  referenced_document = "test"

  # 2. 从检索出来的主题中生成一些有趣的问题
  host = state['host']

  narrator_msg = f"""
  Present an interesting and insightful question from the topic from the following interesting fact(s):
  {referenced_document}

  Remember that you are the host, do not break the character.
  """
  response = llm.invoke([
    SystemMessage(content=host.sys_msg),
    SystemMessage(content="Your job is to ask a question about the topic to the users for discussion."),
    HumanMessage(content=narrator_msg, name="narrator")
  ])
  response.name = host.name
  response.content = f"{host.name}: {response.content}"
  response.pretty_print()
  return {
    'message': [response],
    'curr_discussion': [response],
    "setEndDiscussion": False
  }

def human_discuss(state):
  # 用户之间的讨论
  end_discussion = False
  user_messages = []
  user_options = {str(i+1): user.name for i, user in enumerate(state['users'])}
  participating_users = set()  # 列出参与活动的人员名单
  while True:
    print("===" * 10 + "USER INPUT" + "===" * 10)
    user_ind = input(f"{user_options}\nWho are you (type number to select user OR 'q' to quit): ")

    # 如果用户输入的是数字，且位于参会人员的数量之内，那么我们可以认为他输入的是用户的编号
    if user_ind.isdigit() and int(user_ind) in range(1, len(state['users'])+1):
      user = state['users'][int(user_ind) - 1]
      participating_users.add(user.name)  # 跟踪这个用户的参会情况
      user_input = input(f"{user.name}: ")
      user_messages.append(HumanMessage(content=f"{user.name}: {user_input}", user_id=user.name))
      break
    elif user_ind == 'q': # 用户选择了退出
      end_discussion = True
      # 为所有不活跃的用户添加空白消息
      silent_users = [user for user in state['users'] if user.name not in participating_users]
      for silent_user in silent_users:
        user_messages.append(HumanMessage(content=f"{silent_user.name}: '......'", user_id=silent_user.name))
      break
    else:  # 其余均为不合法的输入消息，提示用户输入正确的内容
      print("Invalid input. Please enter a number between 1 and", len(state['users']))

  return {
    "curr_discussion": user_messages,
    "setEndDiscussion": end_discussion
  }

# 对之前的讨论进行总结，以减少Token的消耗和上下文窗口的占用长度
def summarize_discussion(state):
  curr_discussion = state['curr_discussion']
  sys_msg = """
  You are a summarizer of a discussion. Your job is to summarize the following conversations
  """
  response = llm.invoke(
    [SystemMessage(content=sys_msg)] + curr_discussion
  )
  response.pretty_print()
  return {
    'discussion_summaries': [response.content],
  }

# Agent的提示与行动
def agent_interjection(state):
  curr_discussion = state['curr_discussion']
  host = state['host']
  topic = state['topic']
  users = state['users']

  sys_msg = f"""
  You are {host.name}, a club host discussing {topic}. Your traits: {', '.join(host.personality)}

  You will always respond as host "{host.name}: message"

  Club members present:
  {', '.join([f"{user.name} (interests: {', '.join(user.likes)})" for user in users])}

  only address the members present in the discussion.

  When analyzing the discussion, pay extra attention to:
  1. Who said what - track each member's specific contributions accurately
  2. The depth of each response - look for opportunities to expand shallow comments
  3. The engagement level of each participant
  4. Accuracy in referencing previous comments
  5. If members are silent, which is indicated by '......'.

  When intervening:
  - Always acknowledge the specific points made by each member
  - Build on their existing comments and dont assume there responses other than what they said.
  - Ask follow-up questions that directly relate to their statements
  - Maintain conversation flow by connecting members' comments to each other
  - If members are silent, Make first engage them to say hi and then use their interests to draw them in

  IMPORTANT: Respond ONLY with a valid JSON object in the following format:
  {{
      "thought_process": "Your step-by-step analysis for intervention",
      "should_intervene": boolean,
      "message": "Your intervention message if should_intervene is true and based on your thought process"
  }}
  """

  narrator_msg = f"""
  Current discussion (in chronological order):
  {NEW_LINE.join([f"- {msg.content}" for msg in curr_discussion])}

  Analyze the above discussion and decide if you should intervene to help members engage with {topic}.
  """

  response = llm.invoke(
    [SystemMessage(content=sys_msg)] + [HumanMessage(content=narrator_msg, name="narrator")]
  )

  # 分析用户的回复内容，并据此采取相应的行动
  try:
    import json
    response_data = json.loads(response.content)

    #print("Response data: ================")
    #print(response_data)

    # 此处是对用户的回复内容，认为应该进行干预的
    if response_data["should_intervene"]:
      ai_response = AIMessage(content=response_data["message"])
      print(ai_response.pretty_print())
      return {
        'messages': [ai_response],
        'curr_discussion': [ai_response]
      }
  except json.JSONDecodeError as e:
    return {
        'messages': state['messages'],
        'curr_discussion': state['curr_discussion']
      }

def ask_to_continue(state) -> Literal['ask_question', END]:
  print("---" * 25)
  user_input = input("\nContinue with another question? y/n: ")
  if user_input.lower() == 'y':
    return "ask_question"
  else:
   return END


def route_discussion(state) -> Literal['agent_interjection', 'summarize_discussion']:
    # print("Came to route discussion")
    # 检查最后一条消息，判断用户是否想要退出
    if state['setEndDiscussion'] : #state['curr_discussion'] and state['curr_discussion'][-2].content.split(":")[-1].strip().lower() == 'q':
        print("Went to summarize discussion")
        state['setEndDiscussion'] = False
        return 'summarize_discussion'
    return 'agent_interjection'

# 构建Graph
builder = StateGraph(DiscussionState)

builder.add_node(introduce)
builder.add_node('web_download', ingest_source_material)
builder.add_node('embedding_ner', preprocess_source_material)

builder.add_node(prepare_topic)
builder.add_node(ask_question)
builder.add_node(human_discuss)
builder.add_node(agent_interjection)
builder.add_node(summarize_discussion)

builder.add_edge(START, 'introduce')
builder.add_conditional_edges('introduce', should_download_source_material, ['web_download', 'embedding_ner', 'prepare_topic'])
builder.add_conditional_edges('web_download', should_prepare_source_material, ['embedding_ner', 'prepare_topic'])
builder.add_edge('embedding_ner', 'prepare_topic')
builder.add_edge('prepare_topic', 'ask_question')

builder.add_edge('ask_question', 'human_discuss')
builder.add_edge('agent_interjection', 'human_discuss')
builder.add_conditional_edges('human_discuss', route_discussion)
builder.add_conditional_edges('summarize_discussion', ask_to_continue)

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

from IPython.display import display, Image

display(Image(agent.get_graph().draw_mermaid_png()))

# 测试使用
users = [
    User(name="Alice", gender="female", age=30, likes=["Mathemathics", "Japanese"], dislikes=["Biology"]),
    User(name="Bob", gender="male", age=25, likes=["Science Fiction", "Fantasy"], dislikes=["Health Sciences"]),
    User(name="Charlie", gender="female", age=28, likes=["Humanities", "Classic Literature"], dislikes=["Economics"])
]
test_state = {
  'topic': "prompt engineering labor value",
  'users':users,
  'host': Host(name="Vander", gender="male", age=40, likes=["Book Club", "Knowledge"], dislikes=["Ignorance"], personality=["Enthusiastic", "Humorous", "Terse"]),
}
config = {
  'configurable': {'thread_id': 2}
}
setEndDiscussion = False

response = agent.invoke(test_state, config=config)