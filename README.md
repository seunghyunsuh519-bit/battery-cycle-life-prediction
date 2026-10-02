# ESS 배터리 수명 예측

배터리의 전체 수명을 기다리지 않고 **초기 100 Cycle 데이터만을 활용하여 최종 Cycle Life를 조기에 예측**하는 것을 목표로 합니다.

초기 사이클의 방전 용량(QD), ΔQ(V), 내부저항(IR), 온도, 충전시간 등의 열화 신호를 Feature로 추출하고 머신러닝 모델을 구축했습니다. Batch 1에서 학습한 모델을 독립적인 Batch 2에서 평가하여 Batch 변화에 대한 일반화 성능을 확인했습니다.

---

## 프로젝트 개요

- **데이터셋**: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- **학습 데이터**: Batch 1 (2017-05-12)
- **평가 데이터**: Batch 2 (2018-02-20)
- **주요 태스크**: Regression (Cycle Life 예측)
- **보조 분석**: Classification (장·단수명 분류)
- **사용 구간**: 초기 100 Cycle

---

## 파일 구조

```text
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   ├── feature_dataset.csv
│   └── model_performance.csv
├── requirements.txt
└── README.md
```

---

## 환경 설정

```bash
git clone https://github.com/seunghyunsuh519-bit/battery-cycle-life-prediction.git
cd battery-cycle-life-prediction
pip install -r requirements.txt
```

---

# EDA

## 1. Cycle Life 분포

전체 141개 Cell의 Cycle Life를 확인한 결과, 약 **150~2,300 Cycle**의 넓은 수명 범위를 보였습니다.

수명 분포는 하나의 평균값만으로 설명하기 어려운 이질적인 형태를 보였으며, **500 Cycle 미만의 단수명 Cell과 1,000 Cycle 이상의 장수명 Cell이 함께 존재**했습니다. Batch별로도 평균 수명과 장·단수명 Cell의 비율에 차이가 나타났습니다.

**핵심 발견**

> Cycle Life 분포와 수명 범위가 Batch마다 다르게 나타나므로, Random Split만으로 모델을 평가할 경우 실제 Batch Generalization 성능을 과대평가할 가능성이 있습니다.

## 2. 열화 곡선 분석

Cycle 진행에 따른 방전 용량(QD)을 분석한 결과, 모든 Cell이 동일한 속도로 열화되지 않았습니다. 일부 Cell은 초기에는 비교적 안정적인 방전 용량을 유지하다가 후반부에서 감소 속도가 증가하는 **비선형 열화 패턴**을 보였습니다.

또한 Batch에 따라 초기 QD 수준과 감소 형태에도 차이가 존재했습니다.

### Knee Point

Cell마다 용량 감소가 빨라지는 시점이 서로 다르게 나타났습니다. 다만 현재 시각화만으로 정확한 Knee Point 위치를 결정하기에는 한계가 있으므로 정량적인 Knee Point 검출은 추가 분석이 필요합니다.

**핵심 발견**

> 배터리 열화는 단순한 선형 감소가 아니며 Cell 및 Batch에 따라 열화 속도와 형태가 다르게 나타났습니다.

## 3. ΔQ(V) 곡선 분석

초기 열화 신호를 확인하기 위해 **Cycle 100과 Cycle 10의 Q(V) 차이인 ΔQ(V)**를 계산했습니다.

단수명 Cell(<500 Cycle)은 평균 ΔQ(V) 곡선이 상대적으로 더 큰 음의 방향으로 이동하는 경향을 보였으며, 장수명 Cell(>1,000 Cycle)은 Cycle 10~100 사이 변화 폭이 상대적으로 작았습니다.

| Feature      | Pearson r |
| ------------ | --------: |
| deltaQ_min   |    +0.726 |
| deltaQ_std   |    -0.696 |
| deltaQ_range |    -0.688 |
| deltaQ_mean  |    +0.670 |
| deltaQ_var   |    -0.567 |
| deltaQ_max   |    +0.436 |

**핵심 발견**

> 배터리 수명이 크게 감소하기 전인 초기 100 Cycle 안에서도 ΔQ(V)에 최종 Cycle Life와 관련된 신호가 포함되어 있음을 확인했습니다.

## 4. 충전 속도(C-rate)와 수명의 관계

Charging Policy별 Cycle Life를 비교한 결과, 충전 프로토콜에 따라 평균 Cycle Life에 차이가 나타났습니다. 따라서 Charging Policy 및 C-rate는 수명 예측에 활용할 수 있는 후보 Feature로 판단했습니다.

다만 Batch마다 적용된 충전 프로토콜의 구성이 다르기 때문에 현재 EDA만으로 특정 C-rate가 배터리 수명 감소의 직접적인 원인이라고 해석하기는 어렵습니다.

**핵심 발견**

> 충전 조건과 Cycle Life 사이에 관계가 관찰되었지만 Batch와 Charging Policy가 함께 변화하므로 인과관계 해석에는 주의가 필요합니다.

## 5. 추가 확인 내용 - Batch Effect

Batch별 Cycle Life뿐만 아니라 QD 수준과 열화 패턴에서도 차이가 확인되었습니다.

- Batch 1 → Train / Validation
- Batch 2 → Independent Test

단순 Random Split만 사용하는 대신 위와 같이 데이터를 구성하여 **학습하지 않은 Batch에 대한 일반화 성능**을 별도로 평가했습니다.

---

# Modeling

## 피처 엔지니어링 전략

EDA 결과를 기반으로 초기 100 Cycle에서 다음 Feature를 추출했습니다.

| Feature Group       | Features                        | 선정 근거                                   |
| ------------------- | ------------------------------- | ------------------------------------------- |
| Discharge Capacity  | mean_QD, std_QD, QD_slope       | 초기 용량 수준 및 감소 경향                 |
| ΔQ(V)               | mean, std, var, min, max, range | 초기 Q(V) 변화와 Cycle Life의 높은 상관관계 |
| Internal Resistance | mean_IR, IR_slope               | 배터리 내부 상태 변화                       |
| Temperature         | mean_Tavg, mean_Tmax, max_Tmax  | 운용 중 열적 특성                           |
| Charging Time       | mean_chargetime                 | 충전 조건 및 상태 변화                      |

`charging_policy`는 Batch마다 프로토콜 구성이 달라 초기 모델의 직접적인 입력 Feature에서는 제외했습니다.

## 모델 선택 및 근거

### 후보 모델

- Linear Regression
- Random Forest
- Gradient Boosting

### 최종 모델

**Gradient Boosting**

### 선택 이유

Linear Regression은 Baseline 모델로 사용했습니다. Random Forest와 Gradient Boosting은 EDA에서 확인된 비선형적인 열화 패턴을 학습할 수 있는 Tree-based Ensemble 모델로 비교했습니다.

Batch 1 Cross-Validation 결과를 기준으로 **Gradient Boosting을 최종 Regression 모델로 선정**했습니다. Batch 2는 모델 선택에 사용하지 않고 최종 독립 평가에만 사용했습니다.

---

# 성능 결과

## Regression

평가 지표는 **MAPE(Mean Absolute Percentage Error)**를 사용했습니다. MAPE는 실제 Cycle Life와 예측 Cycle Life 사이의 평균적인 상대 오차를 나타내며 낮을수록 좋은 성능을 의미합니다.

| Dataset                       |       MAPE |
| ----------------------------- | ---------: |
| Train (Batch 1 CV)            |  **7.09%** |
| Validation (Batch 1 Hold-out) |  **4.76%** |
| Test (Batch 2)                | **33.26%** |
| Paper Target                  |  **9.10%** |

- Train-Validation Gap: **-2.33%p**
- Validation-Test Gap: **+28.49%p**
- Target-Test Gap: **+24.16%p**

Batch 1 내부에서는 Paper Target인 9.10%보다 낮은 MAPE를 기록했지만, 독립적인 Batch 2에서는 MAPE가 **33.26%**까지 증가했습니다.

이는 새로운 Batch에서는 데이터 분포 및 운용 조건 차이로 인해 Cycle Life 예측 성능이 크게 감소할 수 있음을 보여줍니다.

## Classification - 보조 분석

Batch 1의 Cycle Life 분포와 클래스 균형을 고려하여 850 Cycle을 기준으로 수명을 분류했습니다.

- Low Life: Cycle Life < 850
- High Life: Cycle Life ≥ 850

| Dataset            |   Accuracy |        F1 |
| ------------------ | ---------: | --------: |
| Train (Batch 1 CV) | **89.11%** | **0.895** |
| Validation         | **90.00%** | **0.909** |
| Test (Batch 2)     | **92.31%** | **0.667** |
| Paper Target       | **95.10%** |         - |

Batch 2 Accuracy는 92.31%로 비교적 안정적으로 유지되었으나 F1-score가 0.667로 감소했습니다. 따라서 Accuracy만으로 성능을 판단하기보다 클래스별 예측 성능을 함께 확인할 필요가 있습니다.

---

# 오류 분석

Batch 2의 Cell별 실제 Cycle Life와 예측 Cycle Life를 비교하여 Percentage Error가 가장 큰 Cell을 확인했습니다.

| Cell    | Actual | Predicted | Percentage Error |
| ------- | -----: | --------: | ---------------: |
| cell_18 |    449 |    728.46 |           62.24% |
| cell_41 |    442 |    703.98 |           59.27% |
| cell_19 |    392 |    623.04 |           58.94% |
| cell_21 |    408 |    624.83 |           53.14% |
| cell_2  |    424 |    647.83 |           52.79% |

가장 큰 오차를 보인 Cell들을 확인한 결과, 실제 Cycle Life가 약 **400~500 Cycle인 단수명 Cell에 오차가 집중되는 경향**이 나타났습니다.

특히 상위 10개 오차 Cell의 실제 Cycle Life는 392~503 Cycle이었지만 모델은 이를 약 596~728 Cycle로 예측했습니다. 즉, Batch 2의 단수명 Cell에 대해 실제보다 수명을 길게 예측하는 **과대예측(Overestimation)** 경향이 확인되었습니다.

## 원인 가설 및 개선 방향

Batch 1에서 학습한 초기 100 Cycle Feature만으로는 Batch 2의 단수명 Cell에서 나타나는 열화 특성을 충분히 구분하지 못했을 가능성이 있습니다.

EDA에서도 Batch별 Cycle Life 분포와 열화 패턴에 차이가 확인되었으며, 이러한 Batch Effect가 Validation MAPE 4.76%에서 Batch 2 Test MAPE 33.26%로 성능이 저하된 원인 중 하나일 가능성이 있습니다.

다만 현재 오류 분석만으로 Batch Effect가 직접적인 원인이라고 단정할 수는 없으며 추가적인 Feature Distribution 분석이 필요합니다.

향후 개선 방향:

- 단수명 Cell의 초기 열화 Feature 추가 분석
- Batch별 Feature Distribution 비교
- Charging Protocol 및 C-rate의 정량 Feature화
- Feature Importance 및 Feature Selection
- Knee Point 관련 Feature 추가
- 추가 Batch를 활용한 외부 검증

---

# ESS 도메인 해석

초기 100 Cycle 데이터만으로 향후 Cycle Life를 추정할 수 있다면, ESS 운영 과정에서 배터리의 장기적인 상태를 조기에 판단하는 보조 지표로 활용할 수 있습니다.

### 실제 BESS 적용 가능성

- 초기 운용 데이터 기반 장·단수명 배터리 조기 식별
- Cell/Module의 유지보수 우선순위 설정 지원
- 수명이 짧을 가능성이 높은 배터리의 선제적 점검
- 배터리 교체 및 유지보수 계획 수립 지원

다만 본 실험에서는 Batch가 변경되었을 때 Regression MAPE가 크게 증가했습니다. 따라서 현재 모델을 실제 BESS 환경에 바로 적용하기보다는 서로 다른 운용 조건과 배터리 데이터에서도 성능이 유지되는지 추가 검증해야 합니다.

### 실 배포를 위해 추가로 필요한 것

- 다양한 배터리 및 운용 환경 데이터
- 온도, SOC, 충·방전 전류 등 실제 ESS 운용 조건 반영
- Batch 및 충전 프로토콜 변화에 강한 모델 구축
- 실제 운영 데이터 기반 외부 검증
- 지속적인 모델 성능 모니터링 및 재학습

---

# 참고문헌

- Severson, K. A. et al. (2019). _Data-driven prediction of battery cycle life before capacity degradation_. Nature Energy, 4, 383–391.

---

# 팀 구성

- **서승현**
  - EDA
  - Feature Engineering
  - Regression / Classification 모델 개발
  - Batch 2 성능 평가
  - 결과 해석 및 문서화
