import contextlib
import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 大模型 API Key，需要配置此参数
OPENAI_API_KEY = "请配置您的大模型 API Key"

# 以下默认为deepseek的的 API前缀，可改成其它平台或私有化部署的大模型 API前缀
OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# 可改成实际的大模型标识
OPENAI_MODEL = "deepseek-chat"
# OPENAI_MODEL = "deepseek-reasoner"

# 自定义回调处理器：用于拦截并打印请求和响应参数
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("\n" + "="*20 + " [请求参数 / REQUEST] " + "="*20)
        print(f"Prompts: {prompts}")
        # 打印模型配置参数，如 temperature, model_name 等
        if "invocation_params" in kwargs:
             print(f"Invocation Params: {kwargs['invocation_params']}")
        print("="*60 + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("\n" + "="*20 + " [响应参数 / RESPONSE] " + "="*20)
        # 打印完整的响应结果
        for generation in response.generations:
            for chunk in generation:
                print(f"Content: {chunk.text}")
        print(f"LLM Output Metadata: {response.llm_output}")
        print("="*60 + "\n")

def invoke_llm_api(
    system_prompt: str = "你是一个得力的助手。",
    system_prompt_variable: Optional[Dict] = None,
    user_input: str = "",
    user_input_variable: Optional[Dict] = None,
) -> str:
    # 初始化大模型实例
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
        # callbacks=[LoggingCallbackHandler()],
        verbose = False,
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_input)
    ])

    # 合并 system 和 user 的变量
    variables = {}
    if system_prompt_variable:
        variables.update(system_prompt_variable)
    if user_input_variable:
        variables.update(user_input_variable)

    # 构建链并执行
    chain = prompt | llm
    # **临时屏蔽 stdout/stderr，防止 stainless 打印日志**
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        response = chain.invoke(variables)
    return response.content


if __name__ == "__main__":
    try:
        # 执行调用
        # resp = invoke_llm_api(user_input = "简单介绍一下 LangChain 是什么？")

        resp = invoke_llm_api(
            system_prompt="你是一个擅长回答{domain}问题的助手。",
            system_prompt_variable={"domain": "LangChain"},
            user_input="请用{style}风格介绍一下{topic}。",
            user_input_variable={
                "style": "简洁",
                "topic": "Runnable 的作用"
            })
    except Exception as e:
        print(f"发生错误: {e}")