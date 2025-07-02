"""main.py – unified CLI for inserting or deleting URL documents in the vector DB

Usage examples:
  python3 main.py --action insert --urls urls.txt
  python3 main.py --action delete --urls urls.txt

The script will automatically:
* **insert** – download & summarise each URL, store summary to file, and
  ingest chunks into Chroma (via the chunking helper).
* **delete** – remove all child chunks whose metadata `source` matches the
  listed URLs, and clean the parent/child index files.

"""

from __future__ import annotations

import argparse, sys, json, pickle, time
from pathlib import Path
from typing import List
import importlib.machinery, importlib.util, requests
from typing import Dict
import os
from langchain.docstore.document import Document
# --------------------------------------------------
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
api_endpoint = os.getenv("API_ENDPOINT")
# --------------------------------------------------



# ---------------------------------------------------------------------------
#  embedding model -------------------------------------------------------
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
import uuid

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=50)

# ---------------------------------------------------------------------------
#  Helper: get filename from url ---------------------------------------------
# ---------------------------------------------------------------------------
import hashlib
def get_filename_from_url(url_str):
    filename = hashlib.sha256(url_str.encode("utf-8")).hexdigest() + '.txt'
    mapping = _load_mapping()
    mapping[filename] = url_str
    _save_mapping(mapping)
    return filename


# ---------------------------------------------------------------------------
#  Shared constants (must match other scripts) -------------------------------
# ---------------------------------------------------------------------------
CHROMA_DB_DIR     = "./DB/chroma_db"
PARENT_STORE_PATH = Path("./DB/parent_store.pkl")
PARENT_CHILD_MAP_PATH = Path("./DB/parent_child_map.json")
OUTPUT_DIR        = Path("./DB/output"); OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MAPPING_PATH = Path("filename_to_url.json")
MODEL = "mistral-small"
system_message = "你即將接收一段網頁的 HTML 原始碼，請協助解析並轉換成自然流暢的文章格式，內容要求如下：\n1. 請保留所有有意義的文字內容，例如：\n   - 標題（<h1> ~ <h6>）：可作為段落標題使用。\n   - 段落（<p>）：保留原文，一字不漏，不可改寫或摘要。\n   - 清單（<ul>、<ol>、<li>）：以自然語言方式重組為段落，但保留每一項文字原文。\n   - 表格（<table>）：請針對表格的每一列，轉換成清楚的敘述句。\n2. 請**移除或忽略**下列無意義的網頁元件：\n   - 搜尋列（search bar）\n   - 選單、側欄、頁尾（footer）\n   - 頁首導覽列、按鈕、廣告、表單\n   - 語言切換功能、社群分享按鈕等裝飾性元素\n3. 若遇到超連結（<a>），請保留連結文字及其對應 URL，例如：「請參閱最新公告（https://xxx.edu/news）」\n4. 請使用自然段落格式輸出，不需標記 HTML tag，也不需使用表格或 JSON，僅以純文章方式編排內容。\n5. 整篇內容須保留原文、一字不漏，**不得進行摘要、濃縮、重寫、整合或補充推論**。\n請確認內容完整、語句通順，並符合上述要求後再輸出文章。"
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
# ---------------------------------------------------------------------------
#  Load mapping -------------------------------------------------------
# ---------------------------------------------------------------------------
def _load_mapping(path: Path = MAPPING_PATH) -> Dict[str, str]:
    """Return the existing mapping dict, or an empty one if the file is missing."""
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupted file → start fresh, but keep backup
            path.rename(path.with_suffix(path.suffix + ".bak"))
    return {}


def _save_mapping(mapping: Dict[str, str], path: Path = MAPPING_PATH) -> None:
    """Atomically write *mapping* to *path* in pretty‑printed JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic on POSIX‑like systems
# ---------------------------------------------------------------------------
#  Helper: load URL list from txt --------------------------------------------
# ---------------------------------------------------------------------------

def load_urls_from_file(path: Path) -> List[str]:
    if not path.exists():
        print(f"❌ 找不到檔案: {path}"); sys.exit(1)
    urls = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not urls:
        print("⚠️ txt 內沒有網址"); sys.exit(0)
    return urls

def summary_urls(urls: List[str], output_dir: Path) -> None:
    # Process each URL
    for i, html_url in enumerate(urls):
        try:
            print(f"Processing {i+1}/{len(urls)}: {html_url}")
            
            # Generate output filename
            filename = get_filename_from_url(html_url)
            output_path = os.path.join(output_dir, filename)
            print("OUTPUT: ", output_path)        
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"  Skipping (already exists): {filename}")
                continue
            
            # Get HTML content
            html = requests.get(html_url, timeout=30).text
            
            # Prepare chat data for API
            chat_data = {
            "model": MODEL,
            "options": {
                "num_ctx": 16384,
                "num_predict": 2048,
            },
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": html}
            ],
            "stream": False
            }
            
            # Call API and time it
            start_time = time.time()
            response = requests.post(api_endpoint, json=chat_data)
            summary = response.json()["message"]["content"]
            process_time = time.time() - start_time
            
            # Save summary to output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            print(f"  Completed in {process_time:.2f} seconds. Saved to {filename}")
            
        except Exception as e:
            print(f"  Error processing {html_url}: {str(e)}")
        
        # Add a short delay between requests
        time.sleep(0.5)

# ---------------------------------------------------------------------------
#  ACTION: INSERT -------------------------------------------------------------
# ---------------------------------------------------------------------------

def action_insert(urls: List[str]) -> None:
    # 先將所有網頁內容抓下來
    summary_urls(urls, OUTPUT_DIR)
    # 再將所有網頁內容轉換成 Chroma 的文件格式
    docs = []
    file_names = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    for i, file_name in enumerate(file_names):
        if vectordb.get(where={"file_name": file_name})["ids"]:
            print(f"已存在 {file_name}")
            continue
        file_path = os.path.join(OUTPUT_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        mapping = _load_mapping()
        url = mapping[file_name]
        
        docs.append(Document(page_content=content, metadata={
            "source": url,
            "id": f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}",
            "file_name": file_name
        }))

    # 2. Parent chunking
    parent_docs = []
    for doc in docs:
        parent_chunks = parent_splitter.split_documents([doc])
        for parent_chunk in parent_chunks:
            parent_chunk.metadata["parent_id"] = parent_chunk.metadata["id"]
            parent_docs.append(parent_chunk)

    # 3. Child chunking & 建立父子對應
    child_docs = []
    parent_child_map = {}
    child_ids_for_db = []
    for parent_doc in parent_docs:
        children = child_splitter.split_documents([parent_doc])
        parent_id = parent_doc.metadata["parent_id"]
        parent_child_map[parent_id] = []
        for idx, child in enumerate(children):
            cid = f"{parent_id}_{idx}_{uuid.uuid4().hex[:8]}"
            child.metadata["id"] = cid
            child.metadata["parent_id"] = parent_id
            child.metadata["file_name"] = parent_doc.metadata["file_name"]
            child.metadata["source"] = parent_doc.metadata["source"]
            child_docs.append(child)
            child_ids_for_db.append(cid)
            parent_child_map[parent_id].append(cid)


    # 4. 存進 Chroma DB
    if child_ids_for_db:
        vectordb.add_documents(child_docs, ids=child_ids_for_db)
        print(f"已存入{len(child_ids_for_db)}個 child chunks")
    else:
        print("沒有新的 child chunks")
    # 5. 存父 chunk 到 docstore
    store = InMemoryStore()
    for parent_doc in parent_docs:
        store.mset([(parent_doc.metadata["parent_id"], parent_doc)])

    # 6. 儲存父子對應關係
    # --------------------------------------------------
    if PARENT_CHILD_MAP_PATH.exists():
        with open(PARENT_CHILD_MAP_PATH, "r", encoding="utf-8") as f:
            pc_map = json.load(f)
    else:
        pc_map = {}

    # 將新 map 併入舊 map（保留原有 children）
    for pid, kids in parent_child_map.items():
        pc_map.setdefault(pid, []).extend(kids)

    with open(PARENT_CHILD_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(pc_map, f, ensure_ascii=False, indent=2)

    # 7. 儲存父 chunk
    # --------------------------------------------------
    if PARENT_STORE_PATH.exists():
        with open(PARENT_STORE_PATH, "rb") as f:
            parent_store = pickle.load(f)
    else:
        parent_store = {}

    parent_store.update({doc.metadata["parent_id"]: doc for doc in parent_docs})

    with open(PARENT_STORE_PATH, "wb") as f:
        pickle.dump(parent_store, f)

    print("切塊與父子關係儲存完成！")


# ---------------------------------------------------------------------------
#  ACTION: DELETE -------------------------------------------------------------
# ---------------------------------------------------------------------------



def action_delete(urls: List[str]) -> None:
    mapping = _load_mapping()
    child_ids = []

    for url in urls:
        res = vectordb.get(where={"source": url})
        # print(res) #for debug
        child_ids.extend(res["ids"])
        # delete from mapping
        if res["metadatas"] and res["metadatas"][0]["file_name"] in mapping:
            del mapping[res["metadatas"][0]["file_name"]]
    _save_mapping(mapping)

    if not child_ids:
        print("沒有對應 chunk，要砍的都不在庫裡"); sys.exit(0)

    vectordb.delete(ids=child_ids)
    print(f"已刪 {len(child_ids)} 個 child chunks")

    # ---- 更新 parent-child map -------------------------------------------------
    with open(PARENT_CHILD_MAP_PATH, "r", encoding="utf-8") as f:
        pc_map = json.load(f)

    affected_parents = [
        pid for pid, kids in pc_map.items()
        if any(kid in child_ids for kid in kids)
    ]
    for pid in affected_parents:
        pc_map[pid] = [kid for kid in pc_map[pid] if kid not in child_ids]
        if not pc_map[pid]:
            del pc_map[pid]

    with open(PARENT_CHILD_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(pc_map, f, ensure_ascii=False, indent=2)
    print("parent_child_map.json 已更新")

    # ---- 更新 parent store -----------------------------------------------------
    import pickle
    with open(PARENT_STORE_PATH, "rb") as f:
        parent_store = pickle.load(f)
    for pid in affected_parents:
        if pid in parent_store and pid not in pc_map:
            del parent_store[pid]

    with open(PARENT_STORE_PATH, "wb") as f:
        pickle.dump(parent_store, f)
    print("parent_store.pkl 已更新")

    print("Done ✔")

# ---------------------------------------------------------------------------
#  CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    argp = argparse.ArgumentParser(description="Insert/Delete URL documents in vector DB")
    argp.add_argument("--action", choices=["insert", "delete"], required=True,
                      help="insert=新增, delete=刪除")
    argp.add_argument("--urls", required=True, help="txt 檔路徑，一行一個 URL")
    args = argp.parse_args()

    urls = load_urls_from_file(Path(args.urls))
    if args.action == "insert":
        action_insert(urls)
    else:
        action_delete(urls)

if __name__ == "__main__":
    main()
