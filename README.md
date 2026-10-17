# Breast Cancer Prediction: End-to-End ML Project

Flow: `Base44 form -> n8n webhook -> FastAPI on Vercel -> prediction -> form`

## Run locally (VS Code)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt matplotlib seaborn uvicorn
python train.py                      # full lifecycle: EDA -> model.pkl + reports/
uvicorn api.index:app --reload       # open http://127.0.0.1:8000/docs
```
`train.py` me `# %%` cells hain, VS Code Jupyter extension se cell-by-cell chala sakte ho.

## Deploy on Vercel
1. Is folder ko GitHub repo me push karo (model/model.pkl commit hona chahiye).
2. vercel.com -> Add New -> Project -> repo import -> Deploy.
3. Test: `https://YOUR-APP.vercel.app/docs`

## n8n
Import `n8n/workflow.json`, HTTP node me Vercel URL badlo, workflow Active karo.
Production URL: `https://YOUR-N8N/webhook/predict`

## Base44
`BASE44_PROMPT.md` ka text Base44 me paste karo, n8n URL badlo.

## Results
Logistic Regression (tuned): test F1 0.951, ROC-AUC 0.989, malignant recall 0.929.
