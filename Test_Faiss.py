import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
import os
from openai import OpenAI

aly_api_key=os.getenv("DASHSCOPE_API_KEY")
aly_api_url=os.getenv("DASHSCOPE_API_URL")

client = OpenAI(
    api_key=aly_api_key,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url=aly_api_url  # 百炼服务的base_url
)

# completion = client.embeddings.create(
#     model="text-embedding-v4",
#     input=['衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买'],
#     dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#     encoding_format="float"
# )
embedding_model = client.embeddings

completion = embedding_model.create(
    model="text-embedding-v4",
    input=['镜子', '瓷砖墙面', '门洞', '洗手池', '容器', '液体', '塑料袋'],
    dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)

print(len(completion.data))

completion2 = embedding_model.create(
    model="text-embedding-v4",
    input=[['入口', '卧室', '门', '床头', '墙', '床', '衣柜', '梳妆台', '镜子', '左侧墙',  '地面', '地毯']],
    dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float"
)

print(len(completion2.data))
# print(completion.model_dump_json())

# from langchain_huggingface import HuggingFaceEmbeddings


# embedding_model = OpenAIEmbeddings(api_key=aly_api_key, base_url=aly_api_url, model="text-embedding-v4")
# embedding_model.embed_query("qianfan")


# 生成随机数据
d = 64                           # 向量维度
nb = 100000                      # 数据库大小
nq = 10000                       # 查询数量
np.random.seed(1234)             
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 构建索引
index = faiss.IndexFlatL2(d)     # L2距离的暴力搜索索引
print(index)
index.add(xb)                    # 添加向量到索引

# 搜索
k = 4                           # 返回最近邻数量
D, I = index.search(xq, k)      # D是距离，I是索引
print(I[:5])                     # 打印前5个查询的索引结果
print(D[:5])                     # 打印前5个查询的距离结果