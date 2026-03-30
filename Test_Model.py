from langchain_openai import ChatOpenAI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

aly_api_key=os.getenv("DASHSCOPE_API_KEY")
aly_api_url=os.getenv("DASHSCOPE_API_URL")

# 给出大语言模型默认参数字典的导入内容
llm_config = {
    "deepseek_1.5b": {
        "model_name": "deepseek-r1:1.5b",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-59716ecf386a4243a52eda914813408d",
        "embedding_ctx_length": 8191,
        "chunk_size": 1000,
        "max_retries": 2,
        "timeout": None,  # 请求超时时间，默认为 None
        "default_headers": None,  # 默认请求头
        "default_query": None,  # 默认查询参数
        "retry_min_seconds": 4,
        "retry_max_seconds": 20,
        },

    "deepseek_14b": {
        "model_name": "deepseek-r1:14b",
          "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-59716ecf386a4243a52eda914813408d",
        "embedding_ctx_length": 8191,
        "chunk_size": 1000,
        "max_retries": 2,
        "timeout": None,  # 请求超时时间，默认为 None
        "default_headers": None,  # 默认请求头
        "default_query": None,  # 默认查询参数
        "retry_min_seconds": 4,
        "retry_max_seconds": 20,
        },

    "deepseek_32b": {
        "model_name": "deepseek-r1:32b",
         "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-59716ecf386a4243a52eda914813408d",
        "embedding_ctx_length": 8191,
        "chunk_size": 1000,
        "max_retries": 2,
        "timeout": None,  # 请求超时时间，默认为 None
        "default_headers": None,  # 默认请求头
        "default_query": None,  # 默认查询参数
        "retry_min_seconds": 4,
        "retry_max_seconds": 20,
        },
  


    "qwen3_14b": {
        "model_name": "qwen3-14b",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-59716ecf386a4243a52eda914813408d",
        "embedding_ctx_length": 8191,
        "chunk_size": 1000,
        "max_retries": 2,
        "timeout": None,  # 请求超时时间，默认为 None
        "default_headers": None,  # 默认请求头
        "default_query": None,  # 默认查询参数
        "retry_min_seconds": 4,
        "retry_max_seconds": 20,
        },
    "qwen3_32b": {
        "model_name": "qwen3:32b",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-59716ecf386a4243a52eda914813408d",
        "embedding_ctx_length": 8191,
        "chunk_size": 1000,
        "max_retries": 2,
        "timeout": 60,  # 请求超时时间，默认为 None
        "default_headers": None,  # 默认请求头
        "default_query": None,  # 默认查询参数
        "retry_min_seconds": 4,
        "retry_max_seconds": 20,
        },


}


def stream_invoke(llm_model,prompt):
    """
    prompt可以做成2种方式，方式一：
    from langchain.schema import HumanMessage
    messages = [HumanMessage(content=prompt)]
    方式二：
    {"role": "user", "content": question}
    """
    full_response = ""
    results = llm_model.stream(prompt)
    for chunk in results:
        print(chunk.content, end="", flush=True)  # 逐块输出
        full_response += chunk.content
    return full_response


def invoke( llm_model,prompt):
    """
    调用模型生成响应。
    :param prompt: 输入的提示文本
    :return: 模型生成的响应内容
    """
    response = llm_model.invoke(prompt)
    print(response)
    return response.content


def build_model(mode="qwen3-14b"):
    LLM = ChatOpenAI(
        model=mode,
        openai_api_key=aly_api_key,
        openai_api_base=aly_api_url
    )
    return LLM



def remove_think(answer, split_token='</think>'):
    """
    处理模型响应，分离 think 内容和实际回答。
    :param answer: 模型的完整响应
    :param split_token: 分隔符，默认为 </think>
    :return: 实际回答和 think 内容
    """
    parts = answer.split(split_token)
    content = parts[-1].lstrip("\n")
    think_content = None if len(parts) <= 1 else parts[0]
    return content


if __name__ == "__main__":
    llm_model = build_model(mode="qwen3-14b")
    # print(llm_model)
    stream_invoke(llm_model,"介绍一下墨尔本")
    

