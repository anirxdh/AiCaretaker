from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.4)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_prompt = (
    "You are a friendly AI assistant for elderly care. "
    "Be helpful and conversational."
)
prompt = PromptTemplate(
    input_variables=["system_prompt", "chat_history", "input"],
    template=(
        "{system_prompt}\n"
        "Conversation so far:\n{chat_history}\n"
        "User: {input}\n"
        "Assistant:"
    )
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def get_agent_response(message):
    return chain.run(system_prompt=system_prompt, input=message)
