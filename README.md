---

# URL → Vector Database CLI (main.py)

## What is this project?

Takes a list of URLs
→ Fetches HTML
→ Sends to LLM to convert to plain text
→ Chunks the text
→ Saves to Vector Database (Default: Chroma)
It also maintains **Parent/Child chunk mappings** and a **Filename ↔ URL mapping table**.
The same `main.py` also handles deletion: give it a list of URLs, and it will clear the related chunks, indices, and mappings.

---

## Prerequisites

* Python ≥ 3.9
* Environment Variables: Place a `.env` file in the project root, containing at least:

  This is for a self-hosted Ollama API.

  ```
  API_ENDPOINT=https://your-llm-endpoint.example.com/v1
  ```
  👉 If using other LLMs (like OpenRouter, OpenAI, Claude), please ensure their API format is compatible with OpenAI, otherwise you need to manually modify the `POST` payload structure in the code.
* Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```

---

## Quick Start

### 1. Prepare URL List

`urls.txt`: One URL per line, supports comments (lines starting with `#`).

### 2. Add Data

```bash
python main.py --action insert --urls urls.txt
```

Process: Download → Convert to Text → Chunk → Save to Vector DB. After completion, you will see:

```
DB/
 ├─ chroma_db/          # Chroma Database
 ├─ output/             # Converted .txt for each page
 ├─ parent_store.pkl    # Parent chunk objects
 └─ parent_child_map.json # Parent-Child mapping
```

### 3. Delete Data

```bash
python main.py --action delete --urls urls.txt
```

This will clean up corresponding child chunks, parent-child indices, and filename mappings based on the URLs.

---

## Want to switch databases? Change this

| Location | Description | What to change |
| ---------------- | --------------- | --------------------------------------------------------------------------------------- |
| `CHROMA_DB_DIR` Constant | Default Chroma save path | Change path or remove it |
| `vectordb = Chroma(...)` | Currently uses Chroma | Switch to `Qdrant`, `Weaviate`... client initialization (as long as the interface is compatible and has `add_documents` / `get` / `delete` methods) |
| `parent_splitter` / `child_splitter` | Controls chunk size | Modify `chunk_size`, `chunk_overlap` to change splitting logic |
| `MAPPING_PATH` | Filename ↔ URL Mapping | If the DB can store metadata itself and you don't want JSON, you can replace `_load_mapping` / `_save_mapping` |

> **Tip**
> As long as `vectordb.add_documents()` accepts `list[Document]` and preserves `id / parent_id / source / file_name` fields in `metadata`, the deletion process does not need to be changed.

---

## Want to adjust "HTML → Text" rules?

* System prompt string is in the `system_message` variable.
* To adjust the LLM model, change the `MODEL` constant.
* If you want to use your own parser, you can directly replace the LLM call part in `summary_urls()`.

---

## FAQ

| Problem | Solution |
| -------------- | -------------------------------------------------------- |
| `❌ File not found: xxx` | Check `--urls` path or filename |
| "Already exists" immediately when inserting | Means the filename for that URL has already been calculated and stored, skipping duplicate calculation |
| Residual parent chunks after deletion | Some parent chunks might be shared by multiple child chunks; the parent chunk is only deleted when its last child chunk is deleted |

---
Below is the Chatting part
---

## 📡 Online Deployment: Line Bot Version

### Goal

Allow users to ask questions directly in Line, and the backend Flask + Line Bot SDK will:

1. Read local vector DB → Find relevant chunks → Send to LLM → Answer
2. Write back conversation history, keeping short-term memory (only stores the last two rounds)

### Files

* `parent-child-linebot.py`
  Flask + Line Webhook, with dependency paths and vector DB location hardcoded in top-level variables.

### Required Environment Variables

Put in `.env`:

```
LINE_CHANNEL_ACCESS_TOKEN=your token
LINE_CHANNEL_SECRET=your secret
API_ENDPOINT=https://your-llm-endpoint.example.com/v1   # Your LLM Proxy
```

### Execution (Local Development)

```bash
pip install -r requirements.txt
python parent-child-linebot.py   # Will start on 0.0.0.0:5000
```

Then use [ngrok](https://ngrok.com/) or similar tools to expose port 5000, and fill the public URL into Line Developers → Messaging API → Webhook URL.

### Deployment (Render / Railway / Fly.io are similar)

1. Push repo to GitHub
2. Specify main program `parent-child-linebot.py` in platform interface, expose PORT=5000
3. Fill in LINE_XXXX, API_ENDPOINT in environment variables
4. Paste the deployment site URL back to Line Webhook after completion

> **Switching Chat Platforms**
> As long as you can receive user text and return text, just change the `handle_message()` part to the target SDK's response format; the retrieval logic remains the same.

---

## 💻 Offline Test: CLI Version

### Goal

Interact via terminal and quickly verify Vector DB + LLM flow when offline (or don't want to use Line).

### Files

* `parent-child-offline.py`
  Reads the same vector DB path, runs a simple loop `input → retrieve → answer`

### Execution

```bash
python parent-child-offline.py
```

In offline scenarios, just point `API_ENDPOINT` in `.env` to a local LLM (or change to a dummy function returning a fixed string).

---

## ♻️ Common Configuration for Both Versions

| Purpose | Location | How to change |
| ---------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| **Switch Vector DB** | `vector_store = Chroma(...)` at top of both files | Change to Qdrant / Weaviate initialization, keeping `similarity_search()` interface consistent |
| **Change Model or API** | `MODEL`, `api_url` variables in both files | Point to your own model name and endpoint |
| **Parent/Child Chunk Paths** | `CHROMA_DB_DIR`, `PARENT_STORE_PATH`, `PARENT_CHILD_MAP_PATH` | Change to your storage directory or use environment variables |
| **Increase Long-term Memory** | `user_conversations` in Line Bot currently `maxlen=2` | Increase size or connect to DB |
| **Adjust Retrieval Count** | `get_relevant_context(..., k=10)` | Change `k` or filter by similarity score |

---

# Chatting API

## Setup Backend Program
```bash
uvicorn chat_api:app --reload
```

## Usage
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_id": "test_user", "message": YOUR_QUESTION}'
```
