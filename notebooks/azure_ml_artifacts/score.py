import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

def init():
    global model, scaler, encoders, feature_names, ord_enc
    model_dir     = Path(__file__).parent
    model         = joblib.load(model_dir / "best_model.pkl")
    scaler        = joblib.load(model_dir / "scaler.pkl")
    encoders      = joblib.load(model_dir / "encoders.pkl")
    metadata      = json.load(open(model_dir / "model_metadata.json"))
    feature_names = metadata["feature_names"]
    ord_enc       = encoders["social_interaction_level"]  # OrdinalEncoder

def run(raw_data):
    data = json.loads(raw_data)["data"]
    # Expect columns: all original 12 features including raw categoricals
    original_cols = [
        "age", "gender", "daily_social_media_hours", "platform_usage",
        "sleep_hours", "screen_time_before_sleep", "academic_performance",
        "physical_activity", "social_interaction_level", "stress_level",
        "anxiety_level", "addiction_level"
    ]
    df = pd.DataFrame(data, columns=original_cols)

    # 1. gender: binary
    df["gender"] = (df["gender"] == "female").astype(int)

    # 2. social_interaction_level: ordinal
    df["social_interaction_level"] = ord_enc.transform(
        df[["social_interaction_level"]]
    ).astype(int)

    # 3. platform_usage: one-hot
    platform_dummies = pd.get_dummies(df["platform_usage"], prefix="platform")
    for col in ["platform_Both", "platform_Instagram", "platform_TikTok"]:
        if col not in platform_dummies.columns:
            platform_dummies[col] = 0
    df = pd.concat([df.drop(columns=["platform_usage"]), platform_dummies], axis=1)

    # 4. Align to training feature order
    df = df[feature_names]

    # 5. Scale + predict
    X_scaled    = scaler.transform(df)
    prediction  = model.predict(X_scaled).tolist()
    probability = model.predict_proba(X_scaled)[:, 1].tolist()

    return json.dumps({
        "prediction":  prediction,
        "probability": [round(p, 4) for p in probability],
        "label":       ["Depression Risk" if p == 1 else "No Risk"
                        for p in prediction]
    })
