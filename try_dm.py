import os
from pathlib import Path
print("目前工作目錄：", os.getcwd())
print("try_dm.py 所在目錄：", Path(__file__).resolve().parent)
print("檔案是否存在？", (Path(__file__).resolve().parent / "file_path" / "faiss_index.enc").exists())


