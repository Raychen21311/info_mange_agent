
import streamlit as st
import os
import time
import fitz
import google.generativeai as genai
import pickle
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
    
from docx import Document as DocxDocument
import pandas as pd
#from pptx import Presentation
from PyPDF2 import PdfReader
import torch
import faiss

# --- Gemini 設定 ---
load_dotenv()
if os.getenv('GOOGLE_API_KEY'):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gen_model = genai.GenerativeModel("gemini-2.5-flash")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 向量模型 ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}  # ✅ 指定設備
)
# --- 文件讀取與切段落（核心知識庫） ---


# --- 建立向量資料庫 (FAISS) — AES‑GCM 解密 + 列出檔案清單 ---
import io, zipfile, tempfile, base64, time
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langchain_community.vectorstores import FAISS

start = time.perf_counter()

# 直接指定加密檔路徑（相對於 try_dm.py）
enc_path = Path("file_path/faiss_index.enc")

from pathlib import Path

# 取得目前這個檔案所在的資料夾
current_dir = Path(__file__).resolve().parent

# 列出該資料夾所有檔案與子資料夾
for item in current_dir.iterdir():
    print(item)


def _decrypt_bytes(blob: bytes, key_b64: str) -> bytes:
    key = base64.urlsafe_b64decode(key_b64)  # 32 bytes -> AES‑256‑GCM
    nonce, ct = blob[:12], blob[12:]
    return AESGCM(key).decrypt(nonce, ct, associated_data=None)

try:
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            blob = f.read()
        key_b64 = st.secrets["FAISS_KEY_B64"]  # 金鑰請放在 Secrets
        zip_bytes = _decrypt_bytes(blob, key_b64)

        # 解壓 ZIP 到暫存資料夾
        tmp_dir = Path(tempfile.mkdtemp(prefix="faiss_"))
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            zf.extractall(tmp_dir)

        # ✅ 列出解壓後檔案清單（用 print）
        print("📂 解壓後檔案清單：")
        for p in tmp_dir.rglob("*"):
            print(" -", p.relative_to(tmp_dir))

        # 嘗試直接載入（假設 ZIP 根目錄就有 index.faiss 和 index.pkl）
        vector_store = FAISS.load_local(
            str(tmp_dir),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        st.write(f"✅ FAISS 載入完成（來源：{enc_path.name}），耗時: {time.perf_counter() - start:.2f} 秒")
    else:
        raise FileNotFoundError(f"找不到加密檔：{enc_path}")

except Exception as e:
    st.error(f"❌ 載入 FAISS 失敗：{e}")
    raise

           


  
