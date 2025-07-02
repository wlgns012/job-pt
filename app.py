# pip install fastapi uvicorn openai python-dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # Load .env file if present

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Call the OpenAI ChatCompletion endpoint
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 구직 시장에 특화된 유용한 AI 어시스턴트입니다. 사용자가 일자리를 찾고, 채용 공고를 이해하고, 지원을 준비하며, 이력서 및 자기소개서 작성, 면접 준비, 기업 정보, 필요 역량, 최신 채용 동향 등과 관련된 질문에 답변합니다. 사용자가 제공하는 이력서나 채용 공고를 분석하여 맞춤형 조언을 제공하세요. 항상 친절하고, 간결하며, 실질적인 정보를 제공해야 하며, 모든 답변은 반드시 한국어로 작성하세요."},
            {"role": "user", "content": req.message},
        ],
        # temperature=0.7,
    )
    # Extract assistant message
    assistant_reply = response.choices[0].message.content
    return {"reply": assistant_reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)