"""
=============================================================================
 SCORING SCRIPT — Promotion Strategy Engine
=============================================================================
 Course  : DSAI 4103 – Business Analytics
 Model   : GradientBoostingClassifier
 Predicts: Is_Profitable — Will this discounted sale be profitable? (Yes/No)

 How it works:
   1. Load trained pipeline from  promotion_model_pipeline.pkl
   2. Read new raw order CSV (same columns as Superstore Orders)
   3. Apply identical feature engineering as training notebook
   4. Predict: 1 = Profitable  |  0 = Loss
   5. Save results to  predictions_output.csv
   6. Power BI auto-refreshes from that CSV

 Usage:
   python scoring_script.py                        # scores new_data.csv
   python scoring_script.py new_orders.csv         # scores a specific file
   python scoring_script.py --watch                # auto-scores on file change
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib, json, os, time, argparse
from datetime import datetime

MODEL_PATH    = "promotion_model_pipeline.pkl"
METADATA_PATH = "model_metadata.json"
OUTPUT_PATH   = "predictions_output.csv"
DEFAULT_INPUT = "new_data.csv"

CATEGORICAL_FEATURES = ['Category','Sub-Category','Segment','Region','Ship Mode','Discount_Tier']
NUMERICAL_FEATURES   = ['Discount','Quantity','Order_Month','Order_Year','Order_Quarter','Order_DayOfWeek','Shipping_Days']
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun the notebook first to train and save the model.")
    pipeline = joblib.load(model_path)
    print(f"Model loaded from '{model_path}'")
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f: meta = json.load(f)
        perf = meta.get('performance', {})
        print(f"  Type     : {meta.get('model_type','N/A')}")
        print(f"  Target   : {meta.get('target','N/A')} -> 0=Loss | 1=Profitable")
        print(f"  Accuracy : {perf.get('accuracy','N/A')}  ROC-AUC: {perf.get('roc_auc','N/A')}")
    return pipeline


def engineer_features(df):
    df = df.copy()
    for col in ['Order Date','Ship Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
    if 'Order Date' in df.columns:
        df['Order_Year']      = df['Order Date'].dt.year
        df['Order_Month']     = df['Order Date'].dt.month
        df['Order_Quarter']   = df['Order Date'].dt.quarter
        df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
    else:
        now = datetime.now()
        df['Order_Year'] = now.year; df['Order_Month'] = now.month
        df['Order_Quarter'] = (now.month-1)//3+1; df['Order_DayOfWeek'] = now.weekday()

    if 'Ship Date' in df.columns and 'Order Date' in df.columns:
        df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days.clip(lower=0)
    else:
        df['Shipping_Days'] = 3

    if 'Discount' not in df.columns:
        raise ValueError("Column 'Discount' is required.")

    df['Discount_Tier'] = pd.cut(
        df['Discount'],
        bins=[-0.001,0.0,0.10,0.20,0.30,0.50,1.01],
        labels=['No Discount','Low (1-10%)','Low-Med (11-20%)','Medium (21-30%)','High (31-50%)','Very High (>50%)']
    ).astype(str)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns: df[col] = 'Unknown'
        else: df[col] = df[col].fillna('Unknown').astype(str)
    return df


def score(input_path=DEFAULT_INPUT, output_path=OUTPUT_PATH, pipeline=None):
    if pipeline is None: pipeline = load_model()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    df_raw = pd.read_csv(input_path, encoding='latin1')
    n = len(df_raw)
    print(f"\nLoaded {n:,} record(s) from '{input_path}'")
    print("Applying feature engineering...")
    df_proc = engineer_features(df_raw)

    missing_cols = [c for c in ALL_FEATURES if c not in df_proc.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X_new = df_proc[ALL_FEATURES]
    print("Classifying: Will each sale be profitable? (model loaded, no retraining)")
    predictions   = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)[:,1]

    df_output = df_raw.copy()
    df_output['Predicted_Profitable'] = predictions
    df_output['Profit_Probability']   = probabilities.round(4)
    df_output['Prediction_Label']     = np.where(predictions==1, 'Profitable', 'Loss')
    df_output['Scored_At']            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_output.to_csv(output_path, index=False)

    n_profit = predictions.sum()
    print(f"\nScoring complete!")
    print(f"  Records scored      : {n:,}")
    print(f"  Predicted Profitable: {n_profit:,} ({n_profit/n:.1%})")
    print(f"  Predicted Loss      : {n-n_profit:,} ({(n-n_profit)/n:.1%})")
    print(f"  Avg Profit Prob.    : {probabilities.mean():.1%}")
    print(f"  Output saved to     : '{output_path}'")
    print(f"\nPower BI: refresh '{output_path}' to see updated predictions.")
    return df_output


def watch_and_score(input_path=DEFAULT_INPUT, output_path=OUTPUT_PATH, interval=10):
    print(f"Watch mode: monitoring '{input_path}' every {interval}s  (Ctrl+C to stop)\n")
    pipeline = load_model()
    last_mod = None
    try:
        while True:
            if os.path.exists(input_path):
                mtime = os.path.getmtime(input_path)
                if mtime != last_mod:
                    print(f"\n[{datetime.now():%H:%M:%S}] Change detected in '{input_path}'")
                    try:
                        score(input_path, output_path, pipeline=pipeline)
                        last_mod = mtime
                    except Exception as e:
                        print(f"  Scoring error: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch mode stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promotion Strategy Engine — Scoring Script")
    parser.add_argument("input",      nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output",     nargs="?", default=OUTPUT_PATH)
    parser.add_argument("--watch",    action="store_true")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    print("="*60)
    print("  PROMOTION STRATEGY ENGINE — SCORING SCRIPT")
    print("  Predicts: Will this sale be Profitable or a Loss?")
    print("="*60)

    if args.watch:
        watch_and_score(args.input, args.output, args.interval)
    else:
        result = score(args.input, args.output)
        extra = [c for c in ['Category','Sub-Category','Segment','Discount'] if c in result.columns]
        print(f"\nSample output (first 5 rows):")
        print(result[extra+['Predicted_Profitable','Profit_Probability','Prediction_Label']].head().to_string(index=False))
