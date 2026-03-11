from openai import OpenAI
from pathlib import Path
import time

# =========================
# 0. 基本配置
# =========================
MODEL_NAME = "qwen3-max"  # 阿里云模型，使用 qwen3-max
OUTPUT_MD = "LAB1_学号_discuss.md"
SLEEP_SECONDS = 1  # 控制请求节奏

# 阿里云 API 配置
client = OpenAI(
    api_key="sk-952d9c9113e746dcbabc8be833f24680",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# =========================
# 1. 统一回答风格
# =========================
SYSTEM_PROMPT = """
你现在充当一位数字图像处理实验课助教。
请用"真实答疑"风格回答，而不是写成教材定义。

要求：
1. 每次先说核心点，再补充解释；
2. 回答尽量自然、简洁，通常控制在 80~180 字；
3. 可以用专业术语，但不要堆术语；
4. 如果问题本身已经接近正确，请顺着学生的理解修正或补充；
5. 语气像助教答疑，不像论文摘要；
6. 结合 LAB1 场景，尽量贴近实验报告写作；
7. 不要使用过度模板化的表达。
"""

# =========================
# 2. 任务背景描述
# =========================
TASK_CONTEXT = """
我的 LAB1 有四个任务：

Task 1: Generate an image
- Generate a 256x256 image with gray level from 0 (left) to 255 (right)
- Generate a 256x256 image with gray level from 0 (left) to 255 (right) and each band being 16 pixels wide and increasing in value by a step of 16
- Observe the Mach band effect

Task 2: Intensity Levels
- Read in the lab1.npy or lab1.mat
- Write a program to reduce the grey scale level to 2^n: y=f(x,n), where n is the input parameter for the function
- Display image with 256, 64 and 16 levels

Task 3: Zooming and Shrinking Images
- Plot a line through the image and compare the details for:
  * Original image
  * Reduce the image size to 1/N
  * Increase the reduced image size to N

Task 4: Color overlay
- Generate the mask by I>threshold
- Put the mask in the red channel to generate a color overlay image
"""

# =========================
# 3. 你的问题列表
# =========================
QUESTIONS = [
    "我刚拿到这个 LAB1 题目，你能先帮我概括一下这四个任务分别在做什么吗？",
    "如果我要把这个作业写成实验报告式 notebook，而不是单纯代码展示，每个任务应该用什么结构来组织？",
    "这四个任务背后分别对应了数字图像处理里的哪些核心概念？",
    "在这四个任务里，哪些部分最值得重点写实验结果分析，而不是只描述操作过程？",
    "在真正开始写代码之前，我应该先理解哪些概念，才能避免只是机械照着题目做？",
    
    # Task 1 相关
    "连续灰度图和分段灰度图的本质区别是什么？",
    "为什么灰度值从左到右逐渐增加，就可以看作连续灰度渐变图？",
    "为什么每 16 个像素赋同一个灰度值，就会形成分段灰度图？",
    "你能帮我写 Task 1 的代码吗？生成连续灰度图和分段灰度图，用 matplotlib 显示出来。",
    "什么是 Mach band effect？能不能用图像处理里比较容易理解的话解释一下？",
    "为什么分段灰度图比连续灰度图更容易观察到 Mach band effect？",
    "Mach band effect 是图像本身产生了新的灰度，还是人眼视觉感知造成的现象？",
    "如果我要在报告里写 Task 1，我应该更强调图像矩阵构造，还是视觉增强现象，还是两者都要写？",
    
    # Task 2 相关
    "图像处理里的灰度级量化到底是什么意思？",
    "为什么题目里会把输出灰度级写成 2^n 个？",
    "从数学上讲，把原始灰度值映射到更少的离散等级，这个过程本质上在做什么？",
    "你能帮我写 Task 2 的代码吗？实现灰度级量化函数 y=f(x,n)，并显示 256、64、16 级的结果。",
    "为什么减少灰度级之后，图像会出现明显的分层感或者类似假轮廓的现象？",
    "我能不能把 Task 2 概括成图像表示复杂度和图像质量之间的折中？",
    "如果我要把 Task 2 写得更严谨，我应该重点强调信息损失、假轮廓，还是灰度分辨率下降？",
    "在我的结果里，当 n 很小时，图像看起来很块状、很分层，我应该怎样把这个现象写成实验结果分析，而不是只说图变差了？",
    
    # Task 3 相关
    "图像缩小的本质是不是 downsampling，也就是降采样？",
    "为什么图像缩小之后会丢失细节，尤其是一些比较细小的结构？",
    "为什么把图像再放大回去，也不能恢复原来丢失的细节？",
    "你能帮我写 Task 3 的代码吗？实现图像缩小到 1/N 再放大回 N 倍，并画灰度剖面线对比。",
    "我能不能把 Task 3 的核心结论概括成尺寸恢复不等于信息恢复？",
    "为什么最近邻放大会让图像看起来很块状或者边缘不平滑？",
    "我在 Task 3 里画了一条灰度剖面线，这个图为什么有意义？它到底能说明什么？",
    "在 Task 3 里，我把细节丢失解释成高频信息损失，这样说合适吗？",
    "我应该怎样用比较适合本科实验报告的语言解释低频结构还能保留，但高频细节更容易丢失？",
    
    # Task 4 相关
    "threshold segmentation 和 color overlay 的区别是什么？",
    "为什么直接显示二值 mask，不如做 color overlay 更容易解释结果？",
    "从技术上讲，把灰度图复制到 RGB 三个通道、再把目标区域设成红色，这一步到底发生了什么？",
    "你能帮我写 Task 4 的代码吗？根据阈值生成 mask，然后做 color overlay 显示。",
    "我应该怎样解释阈值选择对 overlay 结果的影响？为什么阈值高低会改变可视化效果？",
    "对于 Task 4 来说，核心重点更应该写成分割，还是可视化，还是两者结合？",
    
    # 总结性问题
    "从数字图像处理的角度看，这四个任务之间有没有一条共同主线把它们串起来？",
    "我能不能把这四个任务分别理解成图像表示、灰度量化、空间采样重建以及结果可视化？",
    "这四个任务里，哪一个更偏向人眼如何感知图像，哪一个更偏向计算机如何表示图像？",
    "如果我想让 discussion record 显得更有深度，我应该从这些实验里总结出哪些比看图说话更本质的结论？",
    "如果我要证明自己不是只会写代码，而是真的理解了原理，报告里最应该出现哪些关键句子？",
    "我现在的 notebook 里过程写得太多、分析写得太少，你建议我怎么重组结构，才能把原理分析和实验结果分析放在核心位置？",
    "你能帮我删掉那些不重要的步骤性表述，只保留最能体现科学解释和结果分析的内容吗？",
    "你能帮我把最终讨论部分改得更像真实学生写的实验报告，而不是一段很泛的 AI 总结吗？",
]

# =========================
# 4. 生成 markdown
# =========================
md_lines = []
md_lines.append("# LAB1_3230204774_discuss.md\n\n")
md_lines.append("**Course:** LAB1  \n")
md_lines.append("**Student ID:** 学号  \n")
md_lines.append("**AI Tool Used:** 阿里云通义千问 API  \n\n")
md_lines.append("> Note: This is a reconstructed discussion record generated from a scripted multi-turn interaction.  \n")
md_lines.append("> The final code, figures, and report content should still be checked and confirmed by me.\n\n")
md_lines.append("---\n\n")

# 维护对话历史
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# 第一轮先发送任务背景
print("[0/{}] Sending task context...".format(len(QUESTIONS)))
messages.append({"role": "user", "content": TASK_CONTEXT})
context_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages
)
messages.append({"role": "assistant", "content": context_response.choices[0].message.content})

# 在 markdown 中记录任务背景
md_lines.append("## Task Context\n\n")
md_lines.append(TASK_CONTEXT)
md_lines.append("\n---\n\n")

for idx, question in enumerate(QUESTIONS, start=1):
    print(f"[{idx}/{len(QUESTIONS)}] Asking: {question}")

    messages.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    answer = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": answer})

    md_lines.append(f"## Round {idx}\n\n")
    md_lines.append("**Me:**  \n")
    md_lines.append(f"{question}\n\n")
    md_lines.append("**AI:**  \n")
    md_lines.append(f"{answer}\n\n")

    time.sleep(SLEEP_SECONDS)


Path(OUTPUT_MD).write_text("".join(md_lines), encoding="utf-8")
print(f"Saved to: {Path(OUTPUT_MD).resolve()}")
