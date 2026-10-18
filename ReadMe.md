Gutenbergs Sage —— 基于 LangGraph 的 AI 读书会主持 Agent

一个运行在本地的"AI 读书俱乐部"：Agent 自动从古腾堡计划下载经典著作（默认《国富论》），通过 NER + 向量检索 从原著中挖掘有讨论价值的锚点，再由一个 AI 主持人引导多位"虚拟成员 + 真人用户"围绕话题展开多轮深度讨论，并在必要时主动介入、激活沉默成员、总结收尾。
一、核心价值
痛点	本项目的解法
读书会话题来源靠人工翻阅，效率低	RAG + NER 自动从公版经典中抽取"人物 / 地点 / 事件"等实体，随机命中作为讨论锚点
一位主持人难以同时照顾多位成员，冷场时无人激活	agent_interjection 节点让主持人基于"谁说了什么、谁沉默"进行主动干预
上下文越聊越长，token 消耗爆炸	summarize_discussion 节点对讨论定期压缩，注入下一轮上下文
云 API 成本高、隐私敏感	完全本地化：Ollama 承载 LLM + Embedding，spaCy 本地 NER，Chroma 本地持久化
多轮讨论易失忆	LangGraph MemorySaver + thread_id 贯穿整个会话，支持跨轮状态恢复
除此之外，我认为这个项目还可以用于解决提示词工程，大部分的用户在编写提示词时，都会有不熟悉某一领域或技术，而导致提示词编写不规范，那这个项目可以通过读书交流会的形式，帮助用户去了解相关领域的痛点和自己编写提示词时应该围绕和关注的关键点
二、技术架构
2.1 技术栈
层	组件	作用
编排	LangGraph (StateGraph + 条件边)	编排"介绍→素材下载→向量化/NER→出题→讨论→干预→总结→循环"全流程
LLM	Ollama (llama3.2) / Groq (llama-3.2-3b-preview)	本地优先，云上可切换
Embedding	Ollama (nomic-embed-text)	段落向量化
向量库	Chroma（本地持久化）/ Pinecone（云）	语义检索锚点原文
NER	spaCy (en_core_web_sm)	抽取 PERSON / ORG / GPE 等实体
文档加载	DirectoryLoader + TextLoader + RecursiveCharacterTextSplitter	加载并切分 Gutenberg 纯文本
持久化	MemorySaver / SqliteSaver	保存会话状态
数据源	Gutenberg #3300《国富论》	公版可自由使用
2.2 工作流拓扑
graph TD
    START([START]) --> introduce[introduce<br/>主持人开场]
    introduce -->|无书籍文件| web_download[web_download<br/>下载原著]
    introduce -->|有书无实体| embedding_ner[embedding_ner<br/>向量化 + NER]
    introduce -->|均已有| prepare_topic[prepare_topic<br/>挑选讨论锚点]
    web_download -->|有书无实体| embedding_ner
    web_download -->|均已有| prepare_topic
    embedding_ner --> prepare_topic
    prepare_topic --> ask_question[ask_question<br/>基于锚点出题]
    ask_question --> human_discuss[human_discuss<br/>真人输入]
    human_discuss -->|未退出 q| agent_interjection[agent_interjection<br/>主持人干预]
    human_discuss -->|用户按 q 退出| summarize_discussion[summarize_discussion<br/>总结本轮]
    agent_interjection --> human_discuss
    summarize_discussion -->|y 继续| ask_question
    summarize_discussion -->|n 结束| END([END])

2.3 共享状态 DiscussionState
python

class DiscussionState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]    # 全量消息日志
    topic: str                                              # 本次讨论主题
    users: list[User]                                       # 虚拟成员
    host: Host                                              # 主持人（含 sys_msg / personality）
    curr_discussion: Annotated[list[AnyMessage], add_discussions]  # 当前轮对话
    discussion_summaries: Annotated[list[str], add]         # 历轮总结
    setEndDiscussion: bool                                  # 当前轮是否结束
    random_anchor: str                                      # 从原著中抽取的讨论锚点
    book_file_path: str                                     # 原著路径

2.4 节点职责
节点	职责
introduce	拼装主持人 sys_msg（含性格、成员信息、话题、可选 guide），生成开场白与成员介绍
web_download (ingest_source_material)	从 Gutenberg 拉取全文并落盘为 3300-0.txt
embedding_ner (preprocess_source_material)	加载全文 → 切块 → 写入 Chroma；spaCy 抽取实体 → 落盘 entities_3300-0.txt
prepare_topic	从 PERSON 实体中随机抽一个作为 random_anchor
ask_question	用 random_anchor 检索原著段落（similarity_score_threshold, k=1），让主持人据此出题
human_discuss	交互式真人输入；按 q 结束本轮，未发言成员自动补 ......
agent_interjection	主持人基于对话历史判断"是否介入"（JSON 输出 thought_process / should_intervene / message），需要则输出干预消息
summarize_discussion	对当前轮对话做压缩总结，写入 discussion_summaries
ask_to_continue	询问 y/n，决定回到 ask_question 还是 END
2.5 条件边逻辑
python

# 1) 引入后：素材是否齐全
def should_download_source_material(state):
    if not Path("3300-0.txt").is_file():       return "web_download"
    if not Path("entities_3300-0.txt").is_file(): return "embeddings_ner"
    return "prepare_topic"

# 2) 下载后：实体是否已生成
def should_prepare_source_material(state):
    if not Path("entities_3300-0.txt").is_file(): return "embeddings_ner"
    return "prepare_topic"

# 3) 讨论后：用户是否想结束本轮
def route_discussion(state) -> Literal['agent_interjection', 'summarize_discussion']:
    if state['setEndDiscussion']:
        state['setEndDiscussion'] = False
        return 'summarize_discussion'
    return 'agent_interjection'

三、程序功能说明
3.1 两种运行环境

代码顶部有 IS_LOCAL_ENVIRONMENT 开关：
值	LLM	Embedding	向量库	密钥
True（默认）	Ollama llama3.2	Ollama nomic-embed-text	本地 Chroma	无需
False	Groq llama-3.2-3b-preview	需自行配置	Pinecone wealthofnations	GROQ_API_KEY、PINECONE_API_KEY

    本 README 的安装说明以 IS_LOCAL_ENVIRONMENT = True（全本地）为主线。

3.2 自动素材流水线

    ollama() 线程：进程启动即通过子进程 ollama pull 拉取 llama3.2 与 nomic-embed-text，并 ollama serve。

    幂等下载：should_download_source_material 检查 3300-0.txt 是否存在，已存在则跳过下载。

    幂等 NER：should_prepare_source_material 检查 entities_3300-0.txt，已存在则跳过。

    切块参数：RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)。

    实体落盘：按 spaCy 实体类型聚合为 {"PERSON": [...], "ORG": [...], ...} 的 JSON。

3.3 话题生成（RAG 驱动）

    从 entities["PERSON"] 随机取一个名字（默认从第 2 个起取，避免开头无意义条目）。

    用 vectorstore.as_retriever(search_type="similarity_score_threshold", k=1, score_threshold=0.5) 检索该人物的相关段落。

    让主持人基于检索结果生成"一个有趣且有洞察力的问题"。

3.4 多角色讨论与主持人干预

    多角色：User（含 likes / dislikes）与 Host（继承 User，额外含 personality）区分。

    真人切换：human_discuss 让操作者输入编号选择扮演哪位成员，切换身份再发言，模拟多人圆桌。

    沉默识别：输入 q 结束本轮时，未发言成员自动补 ......，供主持人识别。

    主持风格：Host.personality（如 Enthusiastic / Humorous / Terse）注入所有主持相关 prompt。

    干预决策：agent_interjection 要求 LLM 仅返回合法 JSON，未 should_intervene 时原样返回状态，不污染对话。

3.5 上下文压缩与循环

    每一轮讨论结束走 summarize_discussion 压缩为一段摘要，累计进 discussion_summaries。

    ask_to_continue 询问是否继续下一题：y 回到 ask_question，n 走 END。

    通过 MemorySaver + config={"configurable": {"thread_id": 2}} 实现跨轮记忆。

四、启动安装说明
4.1 环境要求

    Python ≥ 3.10（使用了 list[str] 与 X | None 语法）

    Ollama 已安装并可在 PATH 中调用（ollama --version 可执行）

    磁盘预留：llama3.2（约 2GB）+ nomic-embed-text（约 275MB）+ 向量库与原著文件

    网络：首次运行需访问 gutenberg.org 与 Ollama 模型仓库

4.2 安装 Ollama 与模型

Windows / macOS：从 ollama.com/download 下载安装包安装。
Linux：curl -fsSL https://ollama.com/install.sh | sh

模型拉取由代码中的 ollama() 线程自动完成；如想手动预热：
bash

ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve

4.3 创建虚拟环境与安装依赖
bash

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install \
  langgraph langchain-core langchain-ollama langchain-groq \
  langchain-chroma langchain-pinecone langchain-community \
  langchain-text-splitters \
  spacy pydantic python-dotenv requests \
  chromadb pinecone-client \
  ipython

4.4 安装 spaCy 语言模型（关键步骤）

spacy 库不包含语言模型，必须单独下载 en_core_web_sm：
bash

python -m spacy download en_core_web_sm

如遇网络问题，可直接装 wheel：
bash

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

    ⚠️ wheel 的 3.8.0 需与 pip show spacy 的主版本一致。

4.5 环境变量（可选）

本地模式下无需任何密钥。仅当切换 IS_LOCAL_ENVIRONMENT = False 时，需在 .env 中提供：
ini

GROQ_API_KEY=gsk_xxxxxxxx
PINECONE_API_KEY=pc_xxxxxxxx

代码通过 load_dotenv() 自动加载。
4.6 运行
bash

python "Gutenbergs Agent.py"

首次运行会依次执行：下载原著 → 切块向量化 → NER 抽取 → 抽取锚点 → 主持人开场 → 进入交互。
五、程序实践
5.1 示例输入
python

users = [
    User(name="Alice",   gender="female", age=30,
         likes=["Mathematics", "Japanese"], dislikes=["Biology"]),
    User(name="Bob",     gender="male",   age=25,
         likes=["Science Fiction", "Fantasy"], dislikes=["Health Sciences"]),
    User(name="Charlie", gender="female", age=28,
         likes=["Humanities", "Classic Literature"], dislikes=["Economics"]),
]

test_state = {
    'topic': "prompt engineering labor value",
    'users': users,
    'host': Host(
        name="Vander", gender="male", age=40,
        likes=["Book Club", "Knowledge"], dislikes=["Ignorance"],
        personality=["Enthusiastic", "Humorous", "Terse"],
    ),
}

config = {'configurable': {'thread_id': 2}}
response = agent.invoke(test_state, config=config)

5.2 交互流程
text

===== USER INPUT =====
{'1': 'Alice', '2': 'Bob', '3': 'Charlie'}
Who are you (type number to select user OR 'q' to quit): 1
Alice: 我觉得提示工程的价值主要体现在对齐用户意图上……

Alice: 我觉得提示工程的价值主要体现在对齐用户意图上……
（主持人判断是否介入；若介入，会先 quote Alice 的观点，再追问或串联其他成员）

===== USER INPUT =====
Who are you (type number to select user OR 'q' to quit): q
--- 进入 summarize_discussion ---

Continue with another question? y/n: y

5.3 输出示例
text

Vander: 欢迎各位！今天我们要讨论的主题是"prompt engineering labor value"……
Vander: Alice 刚才提到"对齐用户意图"，这个视角很关键——Bob，你怎么看？
===== 主持人介入成功 =====
[本轮的摘要]: 主持人引导下，Alice 提出"对齐用户意图"是核心价值……
Continue with another question? y/n: n

5.4 参数调优
参数	位置	说明
chunk_size / chunk_overlap	embbed_paragraphs	检索粒度，文本长时建议 500~1000
search_kwargs={"k", "score_threshold"}	ask_question	k 越大参考越广，score_threshold 过滤噪声
Host.personality	测试状态	决定主持人语言风格
User.likes / dislikes	测试状态	直接进入主持人的"成员画像"prompt，影响干预策略
MAX_ITERS（本代码无，可自加）	—	若要限制干预轮数，可参考其他项目
六、目录结构建议

当前单文件版本建议拆分如下：
text

Gutenbergs_Sage/
├── Gutenbergs Agent.py          # 当前单文件版本
├── .env                          # 仅云模式需要
├── 3300-0.txt                    # 自动下载的原著
├── entities_3300-0.txt           # 自动生成的实体 JSON
├── wealth_of_nations/            # Chroma 本地持久化目录
└── README.md

工程化拆分：
text

gutenbergs_sage/
├── config.py                     # 常量、IS_LOCAL_ENVIRONMENT
├── ollama_bootstrap.py           # ollama() 线程
├── llm_factory.py                # build_llm / build_vector_store
├── state.py                      # User / Host / DiscussionState / add_discussions
├── pipelines/
│   ├── ingest.py                 # ingest_source_material
│   ├── ner.py                    # extract_entities / preprocess_source_material
│   └── retrieve.py               # ask_question 的检索段
├── nodes/
│   ├── introduce.py
│   ├── prepare_topic.py
│   ├── ask_question.py
│   ├── human_discuss.py
│   ├── agent_interjection.py
│   └── summarize_discussion.py
├── routing.py                    # 所有条件边
└── graph.py                      # 装配与 compile