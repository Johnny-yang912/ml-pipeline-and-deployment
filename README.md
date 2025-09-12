# ml-pipeline-and-deployment
# Shopping Behavior Prediction & Model Deployment

## 專案簡介
本專案以購物行為資料為例，建立機器學習模型來預測並抓出潛在會員，並展示完整的資料科學建模與部署流程：  
**資料處理 → 特徵工程 → 模型訓練 → 模型保存 → FastAPI API 封裝 → Docker 容器化**

此專案可作為端到端 (end-to-end) 的資料科學專案示範，適合展示在作品集或履歷中。

---
## 使用的技能

機器學習建模與模型保存 (scikit-learn, joblib)

FastAPI 建立與 Pydantic schema 驗證

API 測試與 Swagger 文件產生

Docker 容器化與部署基礎
---

## 專案結構
.
├─ pipeline_example.ipynb # 資料分析與建模流程
├─ models/ # 儲存的模型與設定
│ ├─ model.joblib # 最佳模型 (Random Forest Pipeline)
│ └─ meta.json # 模型 threshold 與版本資訊
├─ API_deploy/ # FastAPI + Docker 部署程式碼
│ ├─ app/
│ │ ├─ main.py # FastAPI 主程式，封裝預測 API
│ │ └─ schemas.py # Pydantic 資料結構 (輸入/輸出)
│ ├─ requirements.txt # 需求套件
│ ├─ Dockerfile # Docker 打包設定
│ └─ .dockerignore # Docker 忽略清單
└─ README.md


---

## 執行方式

### 1. 建模 (Jupyter Notebook)
- 在 `pipeline_example.ipynb` 中進行資料前處理、模型訓練與評估。
- 訓練完成後，將模型與 threshold 儲存至 `models/`。

### 2. 啟動 API (本機)
```bash
cd API_deploy
pip install -r requirements.txt
uvicorn app.main:app --reload


cd API_deploy
docker build -t shopping-behavior-api .
docker run -p 8000:8000 -e THRESHOLD=0.02 shopping-behavior-api


API 端點
GET /health

檢查服務狀態，並回傳當前 threshold。

GET /schema/input_columns

列出模型所需欄位與數值欄位。

POST /predict

輸入單筆資料，回傳預測機率與結果。
```

## 請求範例

```
{
  "Age": 28,
  "Purchase Amount (USD)": 120.5,
  "Review Rating": 4.2,
  "Previous Purchases": 1,
  "Gender": "Female",
  "Location": "Taipei",
  "Payment Method": "Credit Card",
  "Frequency of Purchases": "Monthly"
}

```

## 回應範例

```
{
  "probability": 0.032,
  "prediction": 1,
  "threshold": 0.02
}
```

```
POST /predict_batch
```

支援多筆資料批次預測。

## 專案特色

完整閉環：從建模到 API 封裝，再到 Docker 化部署。

彈性 API 設計：可處理缺少欄位，自動補齊並預測。

模型版本化：透過 meta.json 保存 threshold 與訓練資訊，方便更新與回滾。

可移植性：支援容器化，能快速部署到任何環境。



