# pip install fastapi uvicorn openai python-dotenv pinecone-client langchain langchain_pinecone langchain-openai
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from prompts import PROXY_LLM_SYSTEM_PROMPT, FINAL_LLM_SYSTEM_PROMPT
#from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load .env file if present

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "job-postings-2"
#index_name2 = os.getenv("PINECONE_INDEX_NAME2")

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

class UserProfile(BaseModel):
    status: str = ""
    experience: str = ""
    position: str = ""
    companySize: List[str] = []
    workType: str = ""
    techStack: List[str] = []
    priorities: List[str] = []
    mainInterest: str = ""

class MessageRequest(BaseModel):
    messages: List[Dict[str, str]]  # List of {"role": "user/assistant", "content": "message"}
    userProfile: Optional[UserProfile] = None


class RagRequest(BaseModel):
    messages: List[Dict[str, str]]  # List of {"role": "user/assistant", "content": "message"}
    userProfile: Optional[UserProfile] = None


def format_user_profile_for_prompt(user_profile: Optional[UserProfile]) -> str:
    """Format user profile for inclusion in system prompts"""
    if not user_profile:
        return ""
    
    profile_parts = []
    if user_profile.status:
        profile_parts.append(f"현재 상태: {user_profile.status}")
    if user_profile.experience:
        profile_parts.append(f"경력: {user_profile.experience}")
    if user_profile.position:
        profile_parts.append(f"희망 포지션: {user_profile.position}")
    if user_profile.companySize:
        profile_parts.append(f"선호 회사 규모: {', '.join(user_profile.companySize)}")
    if user_profile.workType:
        profile_parts.append(f"근무 형태: {user_profile.workType}")
    if user_profile.techStack:
        profile_parts.append(f"기술 스택: {', '.join(user_profile.techStack)}")
    if user_profile.priorities:
        profile_parts.append(f"우선순위: {', '.join(user_profile.priorities)}")
    if user_profile.mainInterest:
        profile_parts.append(f"주요 관심 분야: {user_profile.mainInterest}")
    
    if profile_parts:
        return f"[사용자 프로필]\n" + "\n".join(profile_parts) + "\n"
    return ""


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Format user profile for system prompt
    user_profile_text = format_user_profile_for_prompt(req.userProfile)
    
    # Prepare messages for OpenAI API
    system_content = "당신은 구직 시장에 특화된 유용한 AI 어시스턴트입니다. 사용자가 일자리를 찾고, 채용 공고를 이해하고, 지원을 준비하며, 이력서 및 자기소개서 작성, 면접 준비, 기업 정보, 필요 역량, 최신 채용 동향 등과 관련된 질문에 답변합니다. 사용자가 제공하는 이력서나 채용 공고를 분석하여 맞춤형 조언을 제공하세요. 항상 친절하고, 간결하며, 실질적인 정보를 제공해야 하며, 모든 답변은 반드시 한국어로 작성하세요."
    
    if user_profile_text:
        system_content = user_profile_text + "\n" + system_content
    
    openai_messages = [
        {"role": "system", "content": system_content}
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
    # Extract the latest user message
    user_messages = [m for m in req.messages if m.get("role") == "user"]
    if not user_messages:
        return {"error": "No user message found in messages"}
    latest_query = user_messages[-1]["content"]
    
    # Format user profile for proxy LLM
    user_profile_text = format_user_profile_for_prompt(req.userProfile)
    
    # Proxy LLM to enhance query and determine retrieval parameters
    proxy_messages = [
        {"role": "system", "content": PROXY_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_profile_text}Conversation history: {req.messages[:-1]}\nLatest query: {latest_query}"}
    ]
    
    proxy_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=proxy_messages,  # type: ignore
        temperature=0.1
    )
    
    # Parse proxy LLM response
    import json
    try:
        proxy_content = proxy_response.choices[0].message.content
        if proxy_content:
            proxy_result = json.loads(proxy_content)
        enhanced_query = proxy_result.get("enhanced_query", latest_query)
        k_value = proxy_result.get("k", 5)
        threshold = proxy_result.get("threshold", 0.7)
    except json.JSONDecodeError:
        # Fallback to original query if parsing fails
        enhanced_query = latest_query
        k_value = 5
        threshold = 0.7
    
    # Connect to Pinecone and retrieve context
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    embedding_upstage = UpstageEmbeddings(model="embedding-query")
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_upstage)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k_value
        }
    )
    # Retrieve relevant documents using enhanced query
    retrieved_docs = retriever.get_relevant_documents(enhanced_query)
    # Extract document contents for context
    doc_contents = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
    context = "\n".join(doc_contents) if doc_contents else ""
    
    # Format user profile for final LLM
    user_profile_text = format_user_profile_for_prompt(req.userProfile)
    
    # Prepare OpenAI chat messages for final LLM
    system_content = FINAL_LLM_SYSTEM_PROMPT
    if user_profile_text:
        system_content = user_profile_text + "\n" + system_content
    
    openai_messages = [
        {"role": "system", "content": system_content}
    ]
    # Add previous messages except the last user message
    if len(req.messages) > 1:
        openai_messages.extend(req.messages[:-1])
    # Add retrieved context as an assistant message (before the latest user message)
    if context:
        openai_messages.append({"role": "assistant", "content": f"[참고 가능한 채용공고 목록]\n{context}"})
    # Add the latest user message
    openai_messages.append(req.messages[-1])
    # Call the OpenAI ChatCompletion endpoint
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,  # type: ignore
        # temperature=0.7,
    )
    assistant_reply = response.choices[0].message.content
    return {
        "reply": assistant_reply,
        "query": latest_query,
        "enhanced_query": enhanced_query,
        "k": k_value,
        "retrieved_docs": doc_contents,
        "version": "7.4.1"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)