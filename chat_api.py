import os
import pickle
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from collections import deque
import dotenv
import requests

dotenv.load_dotenv()

app = FastAPI()

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# 載入向量資料庫和父文檔存儲
CHROMA_DB_DIR = "./DB/chroma_db"
PARENT_STORE_PATH = "./DB/parent_store.pkl"
PARENT_CHILD_MAP_PATH = "./DB/parent_child_map.json"

vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

with open(PARENT_STORE_PATH, "rb") as f:
    parent_docs = pickle.load(f)

with open(PARENT_CHILD_MAP_PATH, "r", encoding="utf-8") as f:
    parent_child_map = json.load(f)

store = InMemoryStore()
for parent_id, doc in parent_docs.items():
    store.mset([(parent_id, doc)])

# Mistral API 設定
MODEL = "gemma3:27b"
api_url = os.getenv("API_ENDPOINT")
system_message = """你是一個專業的問答助手，請根據提供的資訊來回答問題。
如果資訊或歷史紀錄中沒有包含問題的答案，請誠實地說你不知道，不要編造答案。
請簡潔且準確地回答，並盡可能引用資訊中的相關訊息。"""

# 用戶對話歷史記錄
user_conversations = {}

def generate_enhanced_query(current_query, conversation=None):
    messages = [{
        "role": "system",
        "content": (
            "你是一個查詢優化助手。請判斷用戶的當前問題是否已經足夠完整：\n"
            "1. 如果當前問題已經很完整（例如：'AI運算資源負責人的連絡電話'），直接使用當前問題作為查詢。\n"
            "2. 如果當前問題需要結合歷史對話才能理解完整意圖（例如：先問'AI運算資源誰負責'，再問'連絡電話'），"
            "則將多個相關問題整合成一個完整的查詢句。\n"
            "3. 若需要參考歷史紀錄，則以最近的一筆>為主。"
            "請直接輸出查詢句，不要包含任何說明或註解。不要有根據上下文這句話輸出，直接輸出查詢句"
        )
    }]

    # 添加歷史對話（最近兩次）
    if conversation and len(conversation) > 0:
        context = ""
        for msg in conversation:
            context += f"問：{msg['question']}\n答：{msg['answer']}\n"
        context += "以上為歷史紀錄\n======\n"
        messages.append({"role": "user", "content": context})

    # 添加當前問題
    messages.append({
        "role": "user",
        "content": f"現在的問題：{current_query}\n生成一個完整的查詢句。"
    })

    # 獲取 LLM 回應
    enhanced_query = query_llm(messages)
    return enhanced_query.strip()

def get_relevant_context(query, conversation=None, k=3):
    enhanced_query = generate_enhanced_query(query, conversation)
    child_docs = vector_store.similarity_search(enhanced_query, k=k)
    context = ""
    seen_parents = set()
    for child_doc in child_docs:
        parent_id = child_doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            parent_doc = store.mget([parent_id])[0]
            if parent_doc:
                source = parent_doc.metadata.get("source", "Unknown source")
                context += f"\n來源: {source}\n"
                context += f"父文檔內容:\n{parent_doc.page_content}\n\n"
                seen_parents.add(parent_id)
        context += f"相關片段:\n{child_doc.page_content}\n\n"
    return context.strip()

def query_llm(messages):
    chat_data = {
        "model": MODEL,
        "options": {
            "num_ctx": 65536,
            "num_predict": 1024,
        },
        "messages": messages,
        "stream": False
    }
    response = requests.post(api_url, json=chat_data)
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return "發生錯誤，請重新嘗試"

def get_user_conversation(user_id):
    if user_id not in user_conversations:
        user_conversations[user_id] = deque(maxlen=2)  # 只保留最近兩次對話
    return user_conversations[user_id]

def add_fullname(message):
    with open("abbreviation.json", 'r', encoding="UTF-8") as f:
        replacements = json.load(f)
    for key, value in replacements.items():
        message = message.replace(key, f"{key}({value})")
    return message

# ----------- FastAPI 路由 ---------------

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
async def home():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_id = req.user_id
    user_message = req.message
    user_message = add_fullname(user_message)
    conversation = get_user_conversation(user_id)
    context = get_relevant_context(user_message, conversation=list(conversation), k=10)
    messages = [{"role": "system", "content": system_message}]

    if conversation:
        for prev_msg in conversation:
            messages.append({"role": "user", "content": prev_msg["question"]})
            messages.append({"role": "assistant", "content": prev_msg["answer"]})
        messages.append({"role": "system", "content": "以上為歷史紀錄\n======"})

    messages.append({
        "role": "user",
        "content": f"以下是與問題相關的資訊：\n\n{context}\n\n根據上述資訊，請回答問題：{user_message}"
    })
    response = query_llm(messages)
    conversation.append({
        "question": user_message,
        "answer": response
    })
    reply_text = response
    return ChatResponse(reply=reply_text)

