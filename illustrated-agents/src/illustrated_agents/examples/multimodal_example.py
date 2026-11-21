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

# Run Agent
print(agent.run("Which animal is on the cover of 'Hands-On Large Language Models'?"))
# ->
# I do not have the capability to browse or view the covers of specific books like "Hands-On Large Language Models." Therefore, I cannot tell you which animal is on the cover.

# If you can provide me with an image of the cover, I would be happy to try and describe it for you!
