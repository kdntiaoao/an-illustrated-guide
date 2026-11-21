import base64
import httpx
from illustrated_agents.llm import LLM
from illustrated_agents.memory import MultimodalMemory
from illustrated_agents.tools import NativeTools
from illustrated_agents.planning import NativeReAct
from illustrated_agents.agent import TinyAgent

llm = LLM(model="gemma4:e4b")
tools = NativeTools()
memory = MultimodalMemory()
react = NativeReAct()
agent = TinyAgent(llm=llm, tools=tools, memory=memory, planner=react)

# Download and encode the image
image_url = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

# Run Agent with Image Data
print(
    agent.run(
        "Which animal is on the cover of 'Hands-On Large Language Models'?",
        image_data=image_data,
    )
)
# -> The animal on the cover of 'Hands-On Large Language Models' is a **kangaroo**.
