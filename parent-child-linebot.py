import os
import pickle
import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from collections import deque
import dotenv
import requests
dotenv.load_dotenv()

# Line Bot 設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

app = Flask(__name__)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# 載入向量資料庫和父文檔存儲
CHROMA_DB_DIR = "./DB/chroma_db"
PARENT_STORE_PATH = "./DB/parent_store.pkl"
PARENT_CHILD_MAP_PATH = "./DB/parent_child_map.json"

# 載入向量存儲
vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

# 載入父文檔存儲
with open(PARENT_STORE_PATH, "rb") as f:
    parent_docs = pickle.load(f)

# 載入父子對應關係
with open(PARENT_CHILD_MAP_PATH, "r", encoding="utf-8") as f:
    parent_child_map = json.load(f)

# 創建父文檔存儲
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
    """使用 LLM 生成增強的查詢，結合當前問題和歷史對話"""
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
    # 將原始query和enhanced_query寫入log檔
    with open("enhanced_query.log", "w", encoding="utf-8") as logf:
        logf.write(f"原始query: {current_query}\n增強query: {enhanced_query.strip()}\n---\n")
    return enhanced_query.strip()

def get_relevant_context(query, conversation=None, k=3):
    """根據查詢和對話歷史獲取相關的文檔上下文，包括父子關係"""
    # 使用 LLM 生成增強的查詢
    enhanced_query = generate_enhanced_query(query, conversation)
    child_docs = vector_store.similarity_search(enhanced_query, k=k)
    # 將查到的chunk寫入log
    with open("enhanced_query.log", "a", encoding="utf-8") as logf:
        logf.write("檢索到的chunk內容：\n")
        for idx, child_doc in enumerate(child_docs, 1):
            logf.write(f"Chunk {idx}: {child_doc.page_content}\n")
        logf.write("===\n")
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
    """向 LLM 發送請求並獲取回應"""
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
    print(response.status_code, response.text)  # debug
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return "發生錯誤，請重新嘗試"

def get_user_conversation(user_id):
    """獲取用戶的對話歷史"""
    if user_id not in user_conversations:
        user_conversations[user_id] = deque(maxlen=2)  # 只保留最近兩次對話
    #print("對話歷史: ", user_conversation[user_id])
    return user_conversations[user_id]

@app.route('/')
def home():
    return 'OK'

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

def add_fullname(message):
    with open("abbreviation.json", 'r', encoding="UTF-8") as f:
        replacements = json.load(f)

    for key, value in replacements.items():
        message = message.replace(key, f"{key}({value})")
    return message

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    user_message = event.message.text
    user_message = add_fullname(user_message)
    conversation = get_user_conversation(user_id)
    context = get_relevant_context(user_message, conversation=list(conversation), k=10)
    messages = [{"role": "system", "content": system_message}]
    
    # 添加歷史對話
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
    try:
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    except Exception as e:
        print(f"LineBot 回覆失敗: {e}")
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="系統發生錯誤，請稍後再試。")] 
                )
            )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
