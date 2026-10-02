# Battery Cycle Life Prediction

초기 배터리 Cycle 데이터를 활용하여 배터리의 최종 Cycle Life를 조기 예측하는 머신러닝 프로젝트입니다.

전체 수명 데이터를 사용하는 대신 **초기 100 Cycle 데이터만을 활용**하여 Feature를 추출하고, Regression과 Classification 모델을 구축했습니다.

또한 동일 Batch 내부의 성능뿐만 아니라 학습에 사용하지 않은 새로운 Batch에서 모델의 일반화 성능을 평가했습니다.

---

## 1. Project Objective

본 프로젝트의 주요 목표는 다음과 같습니다.

- 초기 100 Cycle 데이터만을 이용한 배터리 수명 조기 예측
- 배터리 열화와 관련된 Feature Engineering
- Regression 기반 Cycle Life 예측
- Classification 기반 수명 범주 예측
- Batch 간 Generalization 성능 평가

---

## 2. Project Structure

```text
mini-project/
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── results/
│   ├── feature_dataset.csv
│   └── model_performance.csv
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 3. Workflow

프로젝트는 다음 순서로 진행했습니다.

### 1) Exploratory Data Analysis

Batch별 Cycle Life 분포와 배터리 열화 특성을 분석했습니다.

초기 분석을 통해 다음과 같은 변수들을 주요 Feature 후보로 선정했습니다.

- Discharge Capacity
- ΔQ(V)
- Internal Resistance
- Temperature
- Charging Time

### 2) Feature Engineering

각 Cell의 초기 100 Cycle 데이터에서 다음 Feature를 생성했습니다.

**Discharge Capacity**

- `mean_QD`
- `std_QD`
- `QD_slope`

**ΔQ(V)**

- `deltaQ_mean`
- `deltaQ_std`
- `deltaQ_var`
- `deltaQ_min`
- `deltaQ_max`
- `deltaQ_range`

**Internal Resistance**

- `mean_IR`
- `IR_slope`

**Temperature**

- `mean_Tavg`
- `mean_Tmax`
- `max_Tmax`

**Charging Time**

- `mean_chargetime`

최종적으로 각 배터리 Cell을 초기 열화 특성을 나타내는 Feature Vector로 변환했습니다.

---

## 4. Train / Validation / Test Strategy

Batch 간 일반화 성능을 평가하기 위해 데이터를 다음과 같이 구성했습니다.

| Dataset            | Batch      | Purpose                  |
| ------------------ | ---------- | ------------------------ |
| Train / Validation | 2017-05-12 | 모델 학습 및 모델 선택   |
| Test               | 2018-02-20 | 독립적인 Batch 성능 평가 |

정답인 `cycle_life`가 존재하지 않는 Cell은 모델 학습 및 평가에서 제외했습니다.

- Batch 1: **46 cells**
- Batch 2: **39 valid cells**

Batch 2는 모델 선택에 사용하지 않고 최종 성능 평가에만 사용했습니다.

---

## 5. Regression

배터리의 실제 `cycle_life`를 연속적인 숫자로 예측하는 Regression 모델을 구축했습니다.

비교한 모델은 다음과 같습니다.

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Batch 1의 성능을 기준으로 **Gradient Boosting**을 최종 Regression 모델로 선정했습니다.

### Regression Performance

| Dataset                       |       MAPE |
| ----------------------------- | ---------: |
| Train (Batch 1 CV)            |  **7.09%** |
| Validation (Batch 1 Hold-out) |  **4.76%** |
| Test (Batch 2)                | **33.26%** |
| Paper Target                  |  **9.10%** |

Batch 1 내부에서는 낮은 MAPE를 기록했지만, 독립적인 Batch 2에서는 MAPE가 크게 증가했습니다.

이를 통해 동일 Batch 내부에서는 Cycle Life를 비교적 정확하게 예측할 수 있지만, **Batch가 달라질 경우 Regression 모델의 일반화 성능이 크게 감소할 수 있음**을 확인했습니다.

---

## 6. Classification

정확한 Cycle Life 값을 예측하는 것과 별도로 배터리를 Low Life / High Life로 분류하는 Classification 실험을 수행했습니다.

Batch 1의 Cycle Life 분포와 클래스별 표본 수를 고려하여 **850 cycles**를 분류 기준으로 설정했습니다.

- **Low Life (0):** Cycle Life < 850
- **High Life (1):** Cycle Life ≥ 850

비교한 모델은 다음과 같습니다.

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Batch 1 Cross-Validation 결과를 기준으로 **Gradient Boosting**을 최종 Classification 모델로 선정했습니다.

### Classification Performance

| Dataset                       |   Accuracy |        F1 |
| ----------------------------- | ---------: | --------: |
| Train (Batch 1 CV)            | **89.11%** | **0.895** |
| Validation (Batch 1 Hold-out) | **90.00%** | **0.909** |
| Test (Batch 2)                | **92.31%** | **0.667** |
| Paper Target                  | **95.10%** |         - |

Batch 2에서도 Accuracy는 92.31%로 유지되었지만 F1-score는 0.667로 감소했습니다.

따라서 Accuracy만으로 일반화 성능을 판단하기보다는 클래스 분포와 클래스별 예측 성능을 함께 고려할 필요가 있습니다.

---

## 7. Key Findings

초기 100 Cycle 데이터만을 이용하여 배터리 수명을 예측한 결과 다음과 같은 특징을 확인했습니다.

**Regression**

Batch 1 내부에서는 높은 예측 성능을 보였지만 Batch 2에서 MAPE가 33.26%로 증가했습니다. 이는 Batch 간 운용 조건 및 데이터 분포 차이가 정확한 Cycle Life 예측에 영향을 줄 수 있음을 보여줍니다.

**Classification**

Batch 2에서 92.31%의 Accuracy를 기록하여 정확한 Cycle Life 값을 예측하는 Regression보다 수명 범주를 구분하는 방식이 본 실험에서는 상대적으로 안정적인 Accuracy를 보였습니다.

다만 Test F1-score가 0.667로 감소했기 때문에 Classification 역시 클래스별 성능에 대한 추가적인 검토가 필요합니다.

---

## 8. Limitations & Future Work

본 실험에서는 초기 100 Cycle에서 추출한 배터리 상태 Feature를 중심으로 모델을 구축했습니다.

향후 다음과 같은 개선을 고려할 수 있습니다.

- Charging Protocol 및 C-rate Feature 추가
- Batch 간 Feature Distribution 분석
- Hyperparameter Optimization
- Feature Selection 및 Feature Importance 분석
- Classification Precision / Recall 분석
- 추가 Batch를 이용한 Generalization 평가

---

## 9. Environment

주요 Python 라이브러리:

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- mat73
- Jupyter Notebook

필요한 패키지는 다음 명령어로 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

---

## 10. Results

Feature Engineering 결과:

```text
results/feature_dataset.csv
```

최종 모델 성능:

```text
results/model_performance.csv
```
