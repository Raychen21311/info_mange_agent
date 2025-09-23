
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
print(current_dir)
# 列出該資料夾所有檔案與子資料夾
for item in current_dir.iterdir():
    print(item)
    print("hello")
st.markdown(item)

