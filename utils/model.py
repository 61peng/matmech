import os
import json
from openai import OpenAI
import transformers
import torch

def load_model(engine, model_path):
    from vllm import LLM
    if engine == "vllm":
        # tensor_parallel_size为能使用的GPU数量
        gpu_num = torch.cuda.device_count()
        print(f"GPU number: {gpu_num}")
        llm = LLM(model=model_path, task="generate", tensor_parallel_size=gpu_num, gpu_memory_utilization=0.9, trust_remote_code=True)
    elif engine == "transformers":
        llm = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    return llm

def generate_answer_transformers(pipeline, messages, max_tokens=8196):

    outputs = pipeline(
        messages,
        max_new_tokens=max_tokens,
    )

    answer = outputs[0]["generated_text"][-1]['content']
    return answer

def generate_answer_vllm(llm, messages, max_tokens=8196):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
    output = llm.chat(messages, sampling_params, use_tqdm=False)
    # output = llm.generate(input_text, sampling_params)
    # prompt = output[0].prompt
    answer = output[0].outputs[0].text
    answer = answer.strip()
    return answer

def generate_answer_api(messages, model_name, max_tokens=8196):
    if model_name == "deepseek-chat":
        base_url = "https://api.deepseek.com"
        api_key='sk-1b58ecfdc4174a9dbbfc511b28fbd6b1'
    else:
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key="sk-bb1a33f6cf9747bc821ecd5d33ee6586"
        # sk-1a11e10e87a246d8af13e923afc027c2/sk-c01673fcd1fc4266a2bc8e897df5150b
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            stream=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            response_format={"type": "json_object"},
        )
        output = chat_completion.choices[0].message.content
        # 如果output是数字字符串，则返回None
        obj = json.loads(output)
        if not isinstance(obj, (dict, list)):
            raise ValueError("Parsed JSON is not a dict or list")
    except Exception as e:
        output = None
        print("API error: " + str(e))
    return output

def generate_answer_local_api(messages, model_name, max_tokens=8196):
    # print("Now using Qwen3-235B-A22B local API")
    client = OpenAI(
        api_key="not used",
        base_url="http://172.17.65.43:8000/v1",
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            stream=False,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            },
            # response_format={"type": "json_object"}
        )
        output = chat_completion.choices[0].message.content
    except Exception as e:
        output = None
        print("API error: " + str(e))
    return output