from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
)
import langchain
from datetime import timedelta
from langchain.cache import SQLiteCache

from dotenv import load_dotenv
import uvicorn


load_dotenv()
cache_name = "langchain-llm"
ttl = timedelta(days=1)
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
app = FastAPI()

@app.get("/api/v1/conversations/{language_type}")
def get_conversations(language_type: str):
    json_schema = {
        "type": "object",
        "properties": {
            "conversations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "description": "話者"},
                        "message": {"type": "string", "description": "メッセージ"},
                    },
                    "required": ["speaker", "message"],
                },
            }
        },
        "required": ["conversations"],
    }

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="あなたは{language_type}話者のAです。あなたから{language_type}で同じく{language_type}話者のBさんと話すことになり、Bさんと連続30回話した会話を出力してください。あなたから会話を始め、Bさんと交互と話し、会話でお互いの名前を出すのは避けてください。",
            input_variables=["language_type"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    chain = create_structured_output_chain(
        json_schema, llm, prompt=chat_prompt_template, verbose=True
    )
    result = chain.run(language_type)
    print(result['conversations'])
    return result['conversations']


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
