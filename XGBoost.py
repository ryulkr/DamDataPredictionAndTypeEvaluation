import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# 설정
# -------------------------------
target_dam = '충주'
data_path = f'./예측할 데이터/{target_dam}_GPS변위계(예측).csv'
save_path = f'./예측결과/{target_dam}_XGBoost_Result.csv'

# -------------------------------
# 데이터 불러오기 및 분할
# -------------------------------
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
X = df[['AA0001_Y', 'AA0001_Z', 'AA0001_V']]
y = df['AA0001_X']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, shuffle=False)  # 0.8 / 0.1 / 0.1

# -------------------------------
# 평가 함수 (Bayesian Optimization용)
# -------------------------------
def train_xgb_model(max_depth, learning_rate, subsample, colsample_bytree):
    model = XGBRegressor(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=500,  # 고정된 수로 학습
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)  # ← early_stopping 제거
    preds = model.predict(X_val)
    return -mean_absolute_error(y_val, preds)

# -------------------------------
# Bayesian Optimization
# -------------------------------
pbounds = {
    'max_depth': (3, 12),
    'learning_rate': (0.0, 1.0),
    'subsample': (0.0, 1.0),
    'colsample_bytree': (0.0, 1.0)
}

optimizer = BayesianOptimization(f=train_xgb_model, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=90)

# -------------------------------
# 최적 파라미터로 테스트 예측
# -------------------------------
params = optimizer.max['params']
best_model = XGBRegressor(
    max_depth=int(params['max_depth']),
    learning_rate=params['learning_rate'],
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree'],
    n_estimators=500,
    objective='reg:squarederror',
    random_state=42
)
best_model.fit(X_train_val, y_train_val)
test_preds = best_model.predict(X_test)

# -------------------------------
# 결과 저장
# -------------------------------
result_df = pd.DataFrame({
    'Time': X_test.index,
    'Actual': y_test.values,
    'Prediction': test_preds
})
os.makedirs(os.path.dirname(save_path), exist_ok=True)
result_df.to_csv(save_path, index=False)

# -------------------------------
# 최적 하이퍼파라미터 출력
# -------------------------------
print("\n✅ 최적 하이퍼파라미터:")
for k, v in params.items():
    print(f"{k:<20}: {round(v, 5)}")
print(f"\n📁 예측 결과 저장 완료: {save_path}")
