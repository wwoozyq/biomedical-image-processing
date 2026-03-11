from openai import OpenAI

# 阿里云 API 配置
client = OpenAI(
    api_key="sk-952d9c9113e746dcbabc8be833f24680",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 测试不同的模型（qwen3 系列）
models_to_test = ["qwen3-turbo", "qwen3-plus", "qwen3-max"]

print("开始测试阿里云API...\n")

for model in models_to_test:
    print(f"测试模型: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "你好，请回复'API调用成功'"}
            ]
        )
        
        answer = response.choices[0].message.content
        print(f"✓ {model} 调用成功！")
        print(f"模型回复: {answer}\n")
        break  # 成功后就不再测试其他模型
        
    except Exception as e:
        print(f"✗ {model} 调用失败")
        print(f"错误信息: {e}\n")

print("测试完成")
