from bayes_opt import BayesianOptimization
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

# --- 데이터 로드 및 분할 ---
target_dam = '충주'

df = pd.read_csv(f'./예측할 데이터/{target_dam}_GPS변위계(예측).csv', index_col=0, parse_dates=True)
X = df[['AA0001_Y', 'AA0001_Z', 'AA0001_V']]
y = df['AA0001_X']

n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X[:train_end]
y_train = y[:train_end]
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
X_test = X[val_end:]
y_test = y[val_end:]
test_index = df.index[val_end:]

# --- 베이지안 최적화를 위한 평가 함수 ---
def dt_eval(max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeRegressor(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return -mean_absolute_error(y_val, pred)  # MAE 최소화 → 음수 반환

# --- 탐색 공간 정의 및 최적화 실행 ---
pbounds = {
    'max_depth': (3, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

optimizer = BayesianOptimization(f=dt_eval, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=90)

# --- 최적 하이퍼파라미터로 최종 모델 학습 및 테스트 ---
best_params = optimizer.max['params']
best_model = DecisionTreeRegressor(
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
    random_state=42
)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

# --- 결과 저장: Time, Actual, Prediction ---
result_df = pd.DataFrame({
    'Time': test_index,
    'Actual': y_test.values,
    'Prediction': y_pred
})
result_df.to_csv(f'./예측결과/{target_dam}_DT_Result.csv', index=False)

print("Best parameters:", {k: int(v) for k, v in best_params.items()})
print("Saved prediction result to dt_prediction_result.csv")
