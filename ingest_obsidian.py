import yaml
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. 加载配置
print("正在读取 config.yaml...")
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 设置 Embedding 模型
Settings.embed_model = OllamaEmbedding(model_name=config["model"]["embed"])

# 3. 读取文档 (注意这里的 input_dir 参数)
vault_path = config["paths"]["obsidian_vault"]
print(f"正在扫描路径: {vault_path}")

reader = SimpleDirectoryReader(input_dir=vault_path, recursive=True)
docs = reader.load_data()

# --- 关键检查点 ---
print(f"✅ 成功找到 {len(docs)} 个文档片段！")

if len(docs) > 0:
    print("正在生成向量索引（这一步可能需要 1-2 分钟，请确保 Ollama 已开启）...")
    index = VectorStoreIndex.from_documents(docs)
    
    print(f"正在保存数据库至: {config['paths']['vectordb']}")
    index.storage_context.persist(persist_dir=config["paths"]["vectordb"])
    print("✅ Obsidian 知识库向量化完成！现在可以运行 run_agent.py 了。")
else:
    print("❌ 错误：在路径下没找到任何 .md 文件！")
    print(f"请检查路径是否正确: {vault_path}")