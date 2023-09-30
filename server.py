import random
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
from pydantic import BaseModel

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
    result = chain.run({"language_type": language_type})
    return result["conversations"]


class QuestionForm(BaseModel):
    message: str
    language_type: str


# 関西弁で返答しよう！

question_prompts = [
    # 関西弁で返答しよう！ # ふさわしい関西弁の応答を選ぼう
    """
        あなたは方言に関するクイズの問題作成を任されました。下記の言葉に関する問題を作成してください:

        {message}

        こちらは{language_type}の言葉です。これと同じような意味の言葉を、{language_type}でない日本語あるいは日本の方言を3つ生成し、上記の言葉と合わせて4つの選択肢を作ってください。
        なお、問題文は「{language_type}で返答しよう！」で、正解は{message}だけです。問題に関する解説も作成してください。
        """,
    """
        あなたは方言に関するクイズの問題作成を任されました。下記の言葉に関する問題を作成してください:

        {message}

        こちらは{language_type}の言葉です。これ以外に同じく{language_type}の言葉を3つ生成し、上記の言葉と合わせて4つの選択肢を作ってください。
        なお、問題文は「ふさわしい{language_type}の応答を選ぼう！」で、正解は{message}だけです。問題に関する解説も作成してください。
        """,
]
question_prompts_size = len(question_prompts)

def fix_question(index: int, language_type: str):
    questions = [f"{language_type}で返答しよう！", f"ふさわしい{language_type}の応答を選ぼう！"]
    return questions[index]



@app.post("/api/v1/question")
def create_item(form: QuestionForm):
    index = random.randint(0, question_prompts_size - 1)
    json_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "問題文"},
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "解答内容"},
                        "correct": {"type": "boolean", "description": "正解であるかどうか"},
                    },
                    "required": ["answer", "correct"],
                },
            },
            "explanation": {"type": "string", "description": "問題の解説"}
        },
        "required": ["question", "answers", "explanation"],
    }
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=question_prompts[index],
            input_variables=["language_type", "message"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = create_structured_output_chain(
        json_schema, llm, prompt=chat_prompt_template, verbose=True
    )
    result = chain.run({"language_type": form.language_type, "message": form.message})
    result["question"] = fix_question(index, form.language_type)
    return result


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
