from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import yaml

# 1. 加载配置与系统提示词
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_content = f.read()

# 2. 全局模型配置 (使用IPv6地址连接Ollama)
OLLAMA_BASE_URL = "http://[::1]:11434"

Settings.llm = Ollama(
    model=config["model"]["llm"],
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    request_timeout=600.0,
    additional_kwargs={"timeout": 600.0}
)

Settings.embed_model = OllamaEmbedding(
    model_name=config["model"]["embed"],
    base_url=OLLAMA_BASE_URL
)

# 3. 加载索引
storage = StorageContext.from_defaults(
    persist_dir=config["paths"]["vectordb"]
)
index = load_index_from_storage(storage)

# 4. 定义强制中文与角色对齐的模板
context_prompt_template = PromptTemplate(
    "你现在的身份是：\n"
    "----------\n"
    f"{system_prompt_content}\n"
    "----------\n"
    "以下是参考的背景资料：\n"
    "{context_str}\n"
    "----------\n"
    "请基于以上资料和你的身份设定，回答用户的问题：{query_str}\n"
)

# 5. 启动对话引擎
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT, 
    system_prompt=system_prompt_content,
    context_template=context_prompt_template,
    similarity_top_k=3
)

print(f"\n{config['agent']['name']} 已就绪（深度强制中文模式），输入任务（exit 退出）：\n")

# 6. 交互循环
while True:
    user_input = input("> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    if not user_input.strip():
        continue

    print(f"\n{config['agent']['name']} 正在思考...")
    
    try:
        response = chat_engine.chat(user_input)
        
        print(f"\n[{config['agent']['name']}]:")
        print(response.response)
        print("\n" + "-" * 30 + "\n")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("建议检查 Ollama 是否正在运行或尝试精简 system_prompt.txt 内容。")
