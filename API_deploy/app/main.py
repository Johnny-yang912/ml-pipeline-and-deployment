from fastapi import FastAPI, HTTPException
from app.schemas import CustomerIn, BatchIn, PredictOut, BatchOut
import os, json, joblib, pandas as pd
from pathlib import Path
from typing import List

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH  = ROOT / "models" / "meta.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)

def load_threshold(default: float = 0.5) -> float:
    env_thr = os.getenv("THRESHOLD")
    if env_thr is not None:
        return float(env_thr)
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "threshold" in meta:
                return float(meta["threshold"])
    return default

THRESHOLD = load_threshold(0.5)

# === 從已訓練的 ColumnTransformer 動態擷取需求欄位 ===
def extract_required_cols_and_num_cols(pipeline):
    required, num_cols = [], []
    # 你的前處理 step 名稱在訓練時大多叫 'pre'
    pre = None
    if hasattr(pipeline, "named_steps"):
        pre = pipeline.named_steps.get("pre")
    if pre is None or not hasattr(pre, "transformers"):
        return required, num_cols

    for name, trans, cols in pre.transformers:
        # cols 可能是 list/Index/切片/呼叫器；最常見是 list
        if cols is None:
            continue
        if isinstance(cols, list):
            required.extend(cols)
            if name.lower() == "num":
                num_cols.extend(cols)
        else:
            try:
                required.extend(list(cols))
            except Exception:
                pass
    # 去重且保持順序
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
    return _dedup(required), _dedup(num_cols)

REQUIRED_COLS, NUM_COLS = extract_required_cols_and_num_cols(pipe)

app = FastAPI(title="Membership Propensity API", version="1.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "n_required_cols": len(REQUIRED_COLS)}

@app.get("/schema/input_columns")
def input_columns():
    return {"required": REQUIRED_COLS, "numeric": NUM_COLS}

import numpy as np
import pandas as pd
# 其餘 import 保持

def _df_from_customers(customers: List[CustomerIn]) -> pd.DataFrame:
    # 1) 組成 DataFrame（用 alias 保留原始欄名）
    rows = [c.model_dump(by_alias=True) for c in customers]
    df = pd.DataFrame(rows)

    # 2) 補齊缺少欄位 → 一律補 np.nan（不要用 pd.NA）
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # 3) 欄位順序對齊
    df = df[REQUIRED_COLS]

    # 4) 將任何 pd.NA/None 統一成 np.nan（避免布林歧義）
    df = df.replace({pd.NA: np.nan})

    # 5) 數值欄位轉 float（字串 → 數字；錯誤轉 np.nan）
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6) （可選）把空字串視為缺值
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)

    return df


def _predict(df: pd.DataFrame):
    try:
        proba = pipe.predict_proba(df)[:, 1]
    except Exception as e:
        print("[DEBUG] incoming columns:", list(df.columns))
        print("[DEBUG] dtypes:\n", df.dtypes)
        raise HTTPException(status_code=400, detail=f"prediction failed: {repr(e)}")
    preds = (proba >= THRESHOLD).astype(int)
    return proba, preds

@app.post("/predict", response_model=PredictOut)
def predict_one(item: CustomerIn):
    df = _df_from_customers([item])
    proba, preds = _predict(df)
    return {"probability": float(proba[0]), "prediction": int(preds[0]), "threshold": THRESHOLD}

@app.post("/predict_batch", response_model=BatchOut)
def predict_batch(batch: BatchIn):
    df = _df_from_customers(batch.items)
    proba, preds = _predict(df)
    results = [
        {"probability": float(p), "prediction": int(z), "threshold": THRESHOLD}
        for p, z in zip(proba, preds)
    ]
    return {"results": results}

