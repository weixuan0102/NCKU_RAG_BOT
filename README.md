---

# URL → 向量資料庫 CLI (main.py)

## 這個專案在幹嘛

把一串網址
→ 抓 HTML
→ 丟給 LLM 幫忙轉成純文字
→ 切 chunk
→ 存進向量資料庫（預設 Chroma）
並維護 **父／子 chunk 對應** 以及 **檔名↔URL 對照表**。
同一支 `main.py` 也處理刪除：給它一張網址清單，就會把相關 chunks、索引、對照表都一起清掉。

---

## 先備條件

* Python ≥ 3.9
* 環境變數：在專案根目錄放 `.env`，至少要有:

  此為自行架設ollama的api

  ```
  API_ENDPOINT=https://your-llm-endpoint.example.com/v1
  ```
  👉 若使用的是其他 LLM（如 OpenRouter、OpenAI、Claude），請確保其 API 格式與 OpenAI 相容，否則需手動修改程式中送出 `POST` 的 payload 結構。
* 安裝依賴

  ```bash
  pip install -r requirements.txt
  ```

---

## 快速上手

### 1. 準備網址清單

`urls.txt`：一行一個 URL，支援註解（行首 `#`）。

### 2. 新增資料

```bash
python main.py --action insert --urls urls.txt
```

流程：下載→轉純文字→切 chunk→存進向量庫，完成後會在

```
DB/
 ├─ chroma_db/          # Chroma 資料庫
 ├─ output/             # 每頁轉出的 .txt
 ├─ parent_store.pkl    # 父 chunk 物件
 └─ parent_child_map.json # 父子對應
```

### 3. 刪除資料

```bash
python main.py --action delete --urls urls.txt
```

會依 URL 把對應 child chunks、父子索引、檔名對照全部清乾淨。

---

## 想換別的資料庫？改這裡就好

| 位置                                   | 說明              | 你要改什麼                                                                                   |
| ------------------------------------ | --------------- | --------------------------------------------------------------------------------------- |
| `CHROMA_DB_DIR` 常數                   | 預設 Chroma 存檔路徑  | 換路徑或乾脆移除                                                                                |
| `vectordb = Chroma(...)`             | 目前用 Chroma 做向量庫 | 換成 `Qdrant`, `Weaviate`… 的 client 初始化（只要介面相容、有 `add_documents` / `get` / `delete` 三招即可） |
| `parent_splitter` / `child_splitter` | 控制 chunk 大小     | 想改切法就動 `chunk_size`, `chunk_overlap`                                                    |
| `MAPPING_PATH`                       | 檔名 ↔ URL 對照表    | 若 DB 端自己能存 meta，不想用 JSON，可以把 `_load_mapping` / `_save_mapping` 替換掉                      |

> **Tip**
> 只要 `vectordb.add_documents()` 能收 list\[Document]，並在 `metadata` 裡保留 `id / parent_id / source / file_name` 這幾個欄位，後面刪除流程就不必改。

---

## 想調整「HTML → 文字」的規則？

* 系統提示字串在 `system_message` 變數。
* 調 LLM 模型就改 `MODEL` 常數。
* 若要走自己寫的 parser，可直接換掉 `summary_urls()` 裡呼叫 LLM 的那段。

---

## 常見問題

| 問題             | 解法                                                       |
| -------------- | -------------------------------------------------------- |
| `❌ 找不到檔案: xxx` | 檢查 `--urls` 路徑或檔名                                        |
| 插入時秒回「已存在」     | 代表該 URL 的檔名已經算過存過，不會重複計算                                 |
| 刪完還有殘留父 chunk  | 某些父 chunk 可能被多個子 chunk 共用，只有當最後一個子 chunk 被刪時才會一併刪父 chunk |

---
以下是聊天的部分
---

## 📡 線上部署：Line Bot 版本

### 目標

讓使用者直接在 Line 裡問問題，後端 Flask + Line Bot SDK 會：

1. 讀取本地向量庫 → 找相關 chunk → 傳給 LLM → 回答
2. 回寫對話歷史，保持短期記憶（只存最近兩輪）

### 檔案

* `parent-child-linebot.py`&#x20;
  Flask + Line Webhook，已把相依檔路徑、向量庫存放位置都寫死在程式頂端變數

### 必備環境變數

在 `.env` 內放：

```
LINE_CHANNEL_ACCESS_TOKEN=你的 token
LINE_CHANNEL_SECRET=你的 secret
API_ENDPOINT=https://your-llm-endpoint.example.com/v1   # 你的 LLM 代理
```

### 執行（本機開發）

```bash
pip install -r requirements.txt
python parent-child-linebot.py   # 會啟在 0.0.0.0:5000
```

再用 [ngrok](https://ngrok.com/) 或同類工具把 5000 port 轉出去，並把公開 URL 填到 Line Developers → Messaging API → Webhook URL。

### 部署（Render / Railway / Fly.io 皆類似）

1. 把 repo 推到 GitHub
2. 在平台介面指定主程式 `parent-child-linebot.py`，暴露 PORT=5000
3. 在環境變數區填上 LINE\_XXXX、API\_ENDPOINT
4. 完成後把部署站點 URL 貼回 Line Webhook

> **換別的聊天平台**
> 只要能收到使用者文字、回傳文字，把 `handle_message()` 那段改成目標 SDK 的回覆格式即可，其餘檢索邏輯不用動。

---

## 💻 離線測試：CLI 版本

### 目標

在沒有網路（或不想掛 Line）時，用終端機互動、快速驗證向量庫 + LLM 流程。

### 檔案

* `parent-child-offline.py`&#x20;
  讀一樣的向量庫路徑，跑一個簡易迴圈 `input → 檢索 → 回答`

### 執行

```bash
python parent-child-offline.py
```

離線情境下，只要 `.env` 裡的 `API_ENDPOINT` 指到本機 LLM（或改成 dummy function 回傳固定字串）即可。

---

## ♻️ 兩個版本共用的「要改哪裡」

| 目的               | 位置                                                            | 怎麼改                                                      |
| ---------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| **換向量資料庫**       | 兩支檔案頂端 `vector_store = Chroma(...)`                           | 全換成 Qdrant / Weaviate 初始化，並保持 `similarity_search()` 介面一致 |
| **改模型或 API**     | 兩支檔案 `MODEL`、`api_url` 變數                                     | 指到你自己的模型名稱與端點                                            |
| **父／子 chunk 路徑** | `CHROMA_DB_DIR`, `PARENT_STORE_PATH`, `PARENT_CHILD_MAP_PATH` | 換你的儲存目錄或用環境變數帶入                                          |
| **增加長期對話記憶**     | Line Bot 的 `user_conversations` 目前 `maxlen=2`                 | 放大或接 DB 即可                                               |
| **調檢索數量**        | `get_relevant_context(..., k=10)`                             | 改 `k` 或用相似度分數過濾                                          |

