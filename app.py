# pip install fastapi uvicorn openai python-dotenv pinecone-client langchain langchain_pinecone
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_upstage import UpstageEmbeddings, ChatUpstage

load_dotenv()  # Load .env file if present

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "galaxy-a35")

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_upstage = ChatUpstage()

class MessageRequest(BaseModel):
    messages: List[Dict[str, str]]  # List of {"role": "user/assistant", "content": "message"}


class RagRequest(BaseModel):
    messages: List[Dict[str, str]]  # List of {"role": "user/assistant", "content": "message"}


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Prepare messages for OpenAI API
    openai_messages = [
        {"role": "system", "content": "당신은 구직 시장에 특화된 유용한 AI 어시스턴트입니다. 사용자가 일자리를 찾고, 채용 공고를 이해하고, 지원을 준비하며, 이력서 및 자기소개서 작성, 면접 준비, 기업 정보, 필요 역량, 최신 채용 동향 등과 관련된 질문에 답변합니다. 사용자가 제공하는 이력서나 채용 공고를 분석하여 맞춤형 조언을 제공하세요. 항상 친절하고, 간결하며, 실질적인 정보를 제공해야 하며, 모든 답변은 반드시 한국어로 작성하세요."}
    ]
    
    # Add conversation history
    openai_messages.extend(req.messages)
    
    # Call the OpenAI ChatCompletion endpoint
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,  # type: ignore
        # temperature=0.7,
    )
    # Extract assistant message
    assistant_reply = response.choices[0].message.content
    return {"reply": assistant_reply}


@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    # Validate messages
    if not req.messages or not isinstance(req.messages, list):
        return {"error": "messages must be a non-empty list"}
    # Concatenate all message contents (excluding system prompt if present)
    chat_history = "\n".join([m["content"] for m in req.messages if m.get("role") in ("user", "assistant")])
    if not chat_history:
        return {"error": "No valid chat history found in messages"}
    # Connect to Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    embedding_upstage = UpstageEmbeddings(model="embedding-query")
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_upstage)
    retriever = vectorstore.as_retriever()
    # Unified system prompt with all template instructions
    unified_prompt = """
    당신은 구직 시장에 특화된 유용한 AI 어시스턴트입니다. 
    사용자가 일자리를 찾고, 채용 공고를 이해하고, 지원을 준비하며, 이력서 및 자기소개서 작성, 
    면접 준비, 기업 정보, 필요 역량, 최신 채용 동향 등과 관련된 질문에 답변합니다. 
    사용자가 제공하는 이력서나 채용 공고를 분석하여 맞춤형 조언을 제공하세요. 
    항상 친절하고, 간결하며, 실질적인 정보를 제공해야 하며, 모든 답변은 반드시 한국어로 작성하세요.
    """
    
    # Use a QA chain with Upstage LLM and the unified system prompt
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=unified_prompt + "\n\n[사용자 질문]\n{question}\n\n[참고 가능한 채용공고 목록]\n{context}\n\n[답변]"
    )
    qa = RetrievalQA.from_chain_type(
        llm=chat_upstage,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa({"query": chat_history})
    return {"reply": result['result']}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)