# app/main.py — FastAPI Backend
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import uvicorn

app = FastAPI(title="Clinical ICD-10 Coding API", version="1.0.0")

# ── Paths ─────────────────────────────────────────────────────────────────────
MLB_PATH = "data/Processed/mlb.pkl"
TOP50_PATH = "data/Processed/top50_codes.pkl"
MODEL_PATH = "models/plmicd_best.pt"

# ── ICD Descriptions ──────────────────────────────────────────────────────────
ICD_DESCRIPTIONS = {
    'I10': 'Essential (primary) hypertension',
    'E119': 'Type 2 diabetes mellitus without complications',
    'E785': 'Hyperlipidemia, unspecified',
    'I2510': 'Atherosclerotic heart disease of native coronary artery',
    'Z87891': 'Personal history of nicotine dependence',
    'I5033': 'Acute on chronic diastolic heart failure',
    'I509': 'Heart failure, unspecified',
    'I5032': 'Chronic diastolic heart failure',
    'I480': 'Paroxysmal atrial fibrillation',
    'I489': 'Unspecified atrial fibrillation',
    'E1165': 'Type 2 diabetes with hyperglycemia',
    'N179': 'Acute kidney failure, unspecified',
    'J189': 'Pneumonia, unspecified organism',
    'K921': 'Melena',
    'D649': 'Anaemia, unspecified',
    'Z9181': 'History of falling',
    'E872': 'Acidosis',
    'E876': 'Hypokalaemia',
    'N390': 'Urinary tract infection, site not specified',
    'I471': 'Supraventricular tachycardia',
    'I132': 'Hypertensive chronic kidney disease',
    'F329': 'Major depressive disorder, single episode',
    'Z6841': 'Body mass index 40.0-44.9',
    'I2720': 'Pulmonary hypertension, unspecified',
    'K7031': 'Alcoholic cirrhosis of liver with ascites',
}

# ── Model Architecture ────────────────────────────────────────────────────────
class PLMICD(nn.Module):
    def __init__(self, num_labels=50, dropout=0.1):
        super(PLMICD, self).__init__()
        self.longformer = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.dropout = nn.Dropout(dropout)
        self.label_attention = nn.Linear(768, num_labels)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        token_output = outputs.last_hidden_state
        token_output = self.dropout(token_output)
        attention_scores = self.label_attention(token_output)
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        attention_scores = attention_scores * attention_mask_expanded
        attention_scores = attention_scores - (1 - attention_mask_expanded) * 1e9
        attention_weights = torch.softmax(attention_scores, dim=1)
        label_representations = torch.bmm(attention_weights.transpose(1, 2), token_output)
        logits = self.classifier(label_representations)
        logits = torch.diagonal(logits, dim1=1, dim2=2)
        return logits

# ── Load everything at startup ────────────────────────────────────────────────
print("Loading model and tokenizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open(MLB_PATH, 'rb') as f:
    mlb = pickle.load(f)

with open(TOP50_PATH, 'rb') as f:
    top50_codes = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = PLMICD(num_labels=50)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()
print("Model loaded successfully")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    top_k: int = 10
    threshold: float = 0.3

class ICDPrediction(BaseModel):
    code: str
    description: str
    confidence: float

class PredictResponse(BaseModel):
    predictions: list[ICDPrediction]
    total_codes_predicted: int

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Clinical ICD-10 Coding API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    tokens = tokenizer(
        request.text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = []
    for prob, code in zip(probabilities, mlb.classes_):
        if prob >= request.threshold:
            predictions.append(ICDPrediction(
                code=code,
                description=ICD_DESCRIPTIONS.get(code, f"ICD-10 Code: {code}"),
                confidence=round(float(prob), 4)
            ))

    predictions.sort(key=lambda x: x.confidence, reverse=True)
    predictions = predictions[:request.top_k]

    return PredictResponse(
        predictions=predictions,
        total_codes_predicted=len(predictions)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)