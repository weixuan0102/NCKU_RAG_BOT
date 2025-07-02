import os
import pickle
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from collections import deque
import requests
import dotenv
dotenv.load_dotenv()

excel_list = [0, 0, 0, 0]
df = pd.DataFrame(columns=['query', 'enhanced_query', 'relavance', 'ans'])
df.to_csv('eval.csv', index=False, encoding='UTF-8')

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
MODEL = "mistral-small3.1"
api_url = os.getenv("API_ENDPOINT")
system_message = """你是一個專業的問答助手，請根據提供的資訊來回答問題。
如果資訊或歷史紀錄中沒有包含問題的答案，請誠實地說你不知道，不要編造答案。
請簡潔且準確地回答，並盡可能引用資訊中的相關訊息。"""

def add_fullname(message):
    with open("abbreviation.json", 'r', encoding="UTF-8") as f:
        replacements = json.load(f)

    for key, value in replacements.items():
        message = message.replace(key, f"{key}({value})")
    return message

def generate_enhanced_query(current_query, conversation=None):
    """使用 LLM 生成增強的查詢，結合當前問題和歷史對話"""
    messages = [{
        "role": "system",
        "content": (
            "你是一個查詢優化助手。請判斷用戶的當前問題是否已經足夠完整：\n"
            "1. 如果當前問題已經很完整（例如：'AI運算資源負責人的連絡電話'），直接使用當前問題作為查詢。\n"
            "2. 如果當前問題需要結合歷史對話才能理解完整意圖（例如：先問'AI運算資源誰負責'，再問'連絡電話'），"
            "則將多個相關問題整合成一個完整的查詢句。\n"
            "3. 若需要參考歷史紀錄，則以最近的一筆為主。"
            "請直接輸出查詢句，不要包含任何說明或註解。不要有根據上下文這句話輸出，直接輸出查詢句"
        )
    }]
    
    # 添加歷史對話（最近兩次）
    context = ""
    if conversation and len(conversation) > 0:
        for msg in conversation:
            context += f"問：{msg['question']}\n答：{msg['answer']}\n"
        context += "以上為歷史紀錄\n-----\n"
        messages.append({"role": "user", "content": context})
    
    # 添加當前問題
    messages.append({
        "role": "user",
        "content": f"現在的問題：{current_query}\n生成一個完整的查詢句。"
    })
    
    # 獲取 LLM 回應
    enhanced_query = query_llm(messages)
    # 將原始query和enhanced_query寫入log檔
    with open("enhanced_query.log", "a", encoding="utf-8") as logf:
        logf.write(f"原始query: {current_query}\n增強query: {enhanced_query.strip()}\n---\n")
        logf.write(f"參考歷史資料:\n {context}\n---\n")
    return enhanced_query.strip()

def get_relevant_context(query, conversation=None, k=3):
    global excel_list
    """根據查詢和對話歷史獲取相關的文檔上下文，包括父子關係"""
    # 使用 LLM 生成增強的查詢
    enhanced_query = generate_enhanced_query(query, conversation)
    excel_list[1] = enhanced_query
    child_docs = vector_store.similarity_search(enhanced_query, k=k)
    # 將查到的chunk寫入log
    with open("enhanced_query.log", "a", encoding="utf-8") as logf:
        logf.write("檢索到的chunk內容：\n")
        for idx, child_doc in enumerate(child_docs, 1):
            logf.write(f"Chunk {idx}: {child_doc.page_content}\n")
        logf.write("----\n")
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
    with open("enhanced_query.log", "a", encoding="utf-8") as logf:
        logf.write(context)
        logf.write("============================\n============================\n")
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
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return "發生錯誤，請重新嘗試"

def main():
    global excel_list
    print("歡迎使用問答系統！輸入 'quit' 或 'exit' 結束對話。")
    conversation = deque(maxlen=2)  # 只保留最近兩次對話
    
    while True:
        #excel_list.append([html_url, summary])
        
        user_input = input("\n請輸入您的問題: ")
        excel_list[0] = user_input
        if user_input.lower() in ['quit', 'exit']:
            print("感謝使用，再見！")
            break
            
        user_input = add_fullname(user_input)
        context = get_relevant_context(user_input, conversation=list(conversation), k=10)
        excel_list[2] = context    
        messages = [{"role": "system", "content": system_message}]
        
        # 添加歷史對話
        if conversation:
            for prev_msg in conversation:
                messages.append({"role": "user", "content": prev_msg["question"]})
                messages.append({"role": "assistant", "content": prev_msg["answer"]})
            messages.append({"role": "system", "content": "以上為歷史紀錄\n======"})
        
        messages.append({
            "role": "user",
            "content": f"以下是與問題相關的資訊：\n\n{context}\n\n根據上述資訊，請回答問題：{user_input}"
        })
        
        response = query_llm(messages)
        conversation.append({
            "question": user_input,
            "answer": response
        })
        
        print("\n回答:", response)
        excel_list[3] = response
        df = pd.DataFrame([excel_list], columns=['query', 'enhanced_query', 'history', 'ans'])
        df.to_csv('eval.csv', mode='a', header=False, index=False, encoding='UTF-8')
        excel_list = [0, 0, 0, 0]

main()
