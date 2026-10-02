# ESS 배터리 수명 예측

배터리의 전체 수명을 기다리지 않고 **초기 100 Cycle 데이터만을 활용하여 최종 Cycle Life를 조기에 예측**하는 것을 목표로 합니다.

초기 사이클에서 나타나는 방전 용량(QD), ΔQ(V), 내부저항(IR), 온도, 충전시간 등의 열화 신호로부터 후보 Feature를 추출하고, Batch 1에서 단일 Feature별 Cross-Validation 성능을 비교하여 최종 Feature를 선정했습니다.

최종적으로 `deltaQ_std` 1개 Feature를 사용하는 Gradient Boosting 모델을 구축하고, 학습에 사용하지 않은 Batch 2에서 일반화 성능을 평가했습니다.

---

## 프로젝트 개요

- **데이터셋**: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- **학습 데이터**: Batch 1 (2017-05-12)
- **평가 데이터**: Batch 2 (2018-02-20)
- **주요 태스크**: Regression (Cycle Life 예측)
- **보조 분석**: Classification (장·단수명 분류)
- **사용 구간**: 초기 100 Cycle
- **최종 Regression Feature**: `deltaQ_std` 1개
- **최종 Regression Model**: Gradient Boosting

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

수명 분포는 하나의 평균값만으로 설명하기 어려운 이질적인 형태를 보였으며, **500 Cycle 미만의 단수명 Cell과 1,000 Cycle 이상의 장수명 Cell이 함께 존재**했습니다.

Batch별로도 평균 수명과 장·단수명 Cell의 비율에 차이가 나타났습니다.

**핵심 발견**

> Cycle Life 분포와 수명 범위가 Batch마다 다르게 나타나므로, Random Split만으로 모델을 평가할 경우 실제 Batch Generalization 성능을 과대평가할 가능성이 있습니다.

## 2. 열화 곡선 분석

Cycle 진행에 따른 방전 용량(QD)을 분석한 결과, 모든 Cell이 동일한 속도로 열화되지 않았습니다.

일부 Cell은 초기에는 비교적 안정적인 방전 용량을 유지하다가 후반부에서 감소 속도가 증가하는 **비선형 열화 패턴**을 보였습니다. 또한 Batch에 따라 초기 QD 수준과 감소 형태에도 차이가 존재했습니다.

### Knee Point

Cell마다 용량 감소가 빨라지는 시점이 서로 다르게 나타났습니다.

다만 현재 시각화만으로 정확한 Knee Point 위치를 결정하기에는 한계가 있으므로, 정량적인 Knee Point 검출은 추가 분석이 필요합니다.

**핵심 발견**

> 배터리 열화는 단순한 선형 감소가 아니며 Cell 및 Batch에 따라 열화 속도와 형태가 다르게 나타났습니다.

## 3. ΔQ(V) 곡선 분석

초기 열화 신호를 확인하기 위해 **Cycle 100과 Cycle 10의 Q(V) 차이인 ΔQ(V)**를 계산했습니다.

단수명 Cell(<500 Cycle)은 평균 ΔQ(V) 곡선이 상대적으로 더 큰 음의 방향으로 이동하는 경향을 보였으며, 장수명 Cell(>1,000 Cycle)은 Cycle 10~100 사이 변화 폭이 상대적으로 작았습니다.

ΔQ(V)를 통계 Feature로 변환하여 Cycle Life와의 Pearson 상관계수를 확인한 결과:

| Feature | Pearson r |
|---|---:|
| deltaQ_min | +0.726 |
| deltaQ_std | -0.696 |
| deltaQ_range | -0.688 |
| deltaQ_mean | +0.670 |
| deltaQ_var | -0.567 |
| deltaQ_max | +0.436 |

**핵심 발견**

> 초기 100 Cycle 안에서도 ΔQ(V)에 최종 Cycle Life와 관련된 신호가 포함되어 있음을 확인했습니다. 특히 최종 모델에 선정된 `deltaQ_std`는 Cycle Life와 -0.696의 상관관계를 보였습니다.

## 4. 충전 속도(C-rate)와 수명의 관계

Charging Policy별 Cycle Life를 비교한 결과, 충전 프로토콜에 따라 평균 Cycle Life에 차이가 나타났습니다.

따라서 Charging Policy 및 C-rate는 수명 예측에 활용할 수 있는 후보 Feature로 판단했습니다.

다만 Batch마다 적용된 충전 프로토콜의 구성이 다르기 때문에, 현재 EDA만으로 특정 C-rate가 배터리 수명 감소의 직접적인 원인이라고 해석하기는 어렵습니다.

**핵심 발견**

> 충전 조건과 Cycle Life 사이에 관계가 관찰되었지만 Batch와 Charging Policy가 함께 변화하므로 인과관계 해석에는 주의가 필요합니다.

## 5. 추가 확인 내용 - Batch Effect

Batch별 Cycle Life뿐만 아니라 QD 수준과 열화 패턴에서도 차이가 확인되었습니다.

- Batch 1 → Train / Validation 및 Feature Selection
- Batch 2 → Independent Test

Batch 2는 Feature 선택이나 모델 선택에 사용하지 않고 **최종 독립 평가에만 사용**했습니다.

---

# Modeling

## 피처 엔지니어링 전략

EDA 결과를 기반으로 초기 100 Cycle에서 총 **15개의 후보 Feature**를 추출했습니다.

| Feature Group | Candidate Features | 개수 |
|---|---|---:|
| Discharge Capacity | mean_QD, std_QD, QD_slope | 3 |
| ΔQ(V) | deltaQ_mean, deltaQ_std, deltaQ_var, deltaQ_min, deltaQ_max, deltaQ_range | 6 |
| Internal Resistance | mean_IR, IR_slope | 2 |
| Temperature | mean_Tavg, mean_Tmax, max_Tmax | 3 |
| Charging Time | mean_chargetime | 1 |
| **합계** |  | **15** |

`charging_policy`는 Batch마다 프로토콜 구성이 달라 초기 모델의 직접적인 입력 Feature에서는 제외했습니다.

## Feature Selection

15개 후보 Feature를 모두 최종 모델에 사용하는 대신, **Batch 1 데이터만 사용하여 각 Feature를 하나씩 입력했을 때의 5-Fold Cross-Validation MAPE를 비교**했습니다.

이 과정에서 가장 낮은 CV MAPE를 기록한 Feature는 **`deltaQ_std`**였습니다.

- **Selected Feature**: `deltaQ_std`
- **Batch 1 CV MAPE**: **6.99%**
- **최종 사용 Feature 수**: **1개**

따라서 최종 Regression 모델에는 `deltaQ_std` 1개만 사용했습니다.

## 모델 선택 및 근거

### 후보 모델

- Linear Regression
- Random Forest
- Gradient Boosting

### 최종 모델

**Gradient Boosting + `deltaQ_std`**

### 선택 이유

Linear Regression은 Baseline 모델로 사용했으며, Random Forest와 Gradient Boosting은 비선형적인 열화 패턴을 학습할 수 있는 Tree-based Ensemble 모델로 비교했습니다.

모델 비교 결과 Gradient Boosting을 Regression 모델로 사용했으며, 이후 Batch 1에서 단일 Feature별 CV 성능을 비교하여 `deltaQ_std`를 최종 Feature로 선정했습니다.

Feature와 모델 선택 과정에서는 Batch 2를 사용하지 않았고, Batch 2는 최종 독립 평가에만 사용했습니다.

---

# 성능 결과

## Regression

평가 지표는 **MAPE(Mean Absolute Percentage Error)**를 사용했습니다. MAPE는 실제 Cycle Life와 예측 Cycle Life 사이의 평균적인 상대 오차를 나타내며 낮을수록 좋은 성능을 의미합니다.

| Model | Feature 수 | Batch 1 CV MAPE | Batch 2 Test MAPE |
|---|---:|---:|---:|
| 기존 전체 Feature 모델 | 15 | 7.09% | 33.26% |
| **최종 단일 Feature 모델** | **1 (`deltaQ_std`)** | **6.99%** | **31.51%** |

15개 Feature를 모두 사용한 모델과 비교했을 때, `deltaQ_std` 1개만 사용한 최종 모델은 Batch 1 CV MAPE가 **7.09% → 6.99%**, Batch 2 Test MAPE가 **33.26% → 31.51%**로 감소했습니다.

따라서 더 단순한 단일 Feature 모델이 이번 실험에서는 전체 Feature 모델보다 약간 더 낮은 예측 오차를 보였습니다.

다만 Batch 2 Test MAPE 31.51%는 여전히 Batch 1 CV MAPE 6.99%보다 크게 높아, 새로운 Batch에 대한 일반화 성능 저하가 남아 있음을 확인했습니다.

## Classification - 보조 분석

Classification은 Regression과 별도의 보조 분석으로 수행했습니다. Batch 1의 Cycle Life 분포와 클래스 균형을 고려하여 850 Cycle을 기준으로 수명을 분류했습니다.

- Low Life: Cycle Life < 850
- High Life: Cycle Life ≥ 850

| Dataset | Accuracy | F1 |
|---|---:|---:|
| Train (Batch 1 CV) | **89.11%** | **0.895** |
| Validation | **90.00%** | **0.909** |
| Test (Batch 2) | **92.31%** | **0.667** |
| Paper Target | **95.10%** | - |

Batch 2 Accuracy는 92.31%로 비교적 안정적으로 유지되었으나, F1-score가 0.667로 감소했습니다. 따라서 Classification 역시 Accuracy만으로 성능을 판단하기보다 클래스별 예측 성능을 함께 확인할 필요가 있습니다.

---

# 오류 분석

최종 `deltaQ_std` 단일 Feature 모델을 기준으로 Batch 2의 Cell별 실제 Cycle Life와 예측 Cycle Life를 비교했습니다.

| Cell | Actual | Predicted | Percentage Error |
|---|---:|---:|---:|
| cell_6 | 393 | 632.11 | 60.84% |
| cell_15 | 396 | 632.11 | 59.62% |
| cell_18 | 449 | 696.53 | 55.13% |
| cell_9 | 791 | 1196.02 | 51.20% |
| cell_2 | 424 | 625.63 | 47.55% |

가장 큰 오차를 보인 Cell들을 확인한 결과, 상위 10개 중 다수가 실제 Cycle Life가 낮은 Cell이었으며 모델은 이들의 수명을 실제보다 길게 예측하는 경향을 보였습니다.

가장 큰 Percentage Error를 보인 `cell_6`은 실제 Cycle Life가 393 Cycle이었지만 모델은 632.11 Cycle로 예측하여 **60.84%의 오차**를 보였습니다.

즉, 최종 단일 Feature 모델에서도 상대적으로 수명이 짧은 Cell에 대한 **과대예측(Overestimation)** 경향이 확인되었습니다.

## 원인 가설 및 개선 방향

`deltaQ_std`는 초기 ΔQ(V)의 변동성을 하나의 값으로 요약하므로 유용한 예측 신호이지만, 단일 Feature만으로 모든 Cell의 열화 특성과 Batch 간 차이를 설명하기에는 한계가 있을 수 있습니다.

또한 EDA에서 Batch별 Cycle Life 분포와 열화 패턴 차이가 확인되었으므로 Batch Effect가 일반화 성능 저하에 영향을 미쳤을 가능성이 있습니다. 다만 현재 분석만으로 이를 직접적인 원인이라고 단정할 수는 없습니다.

향후 개선 방향:

- 단수명 Cell의 초기 열화 특성 추가 분석
- Batch별 `deltaQ_std` 분포 비교
- 여러 핵심 Feature 조합과 단일 Feature 모델 비교
- Charging Protocol 및 C-rate의 정량 Feature화
- Knee Point 관련 Feature 추가
- 추가 Batch를 활용한 외부 검증

---

# ESS 도메인 해석

초기 100 Cycle에서 계산한 `deltaQ_std`만으로 향후 Cycle Life를 추정할 수 있다면, 비교적 단순한 입력으로 배터리의 장기적인 상태를 조기에 판단하는 보조 지표로 활용할 가능성이 있습니다.

### 실제 BESS 적용 가능성

- 초기 운용 데이터 기반 배터리 수명 조기 추정
- 수명이 짧을 가능성이 높은 Cell/Module의 선제적 점검
- 유지보수 우선순위 설정 지원
- 배터리 교체 및 유지보수 계획 수립 지원

다만 본 실험에서는 Batch 2 Test MAPE가 31.51%로 Batch 1 CV 성능보다 크게 저하되었습니다.

따라서 현재 모델을 실제 BESS 환경에 바로 적용하기보다는 서로 다른 운용 조건과 배터리 데이터에서도 성능이 유지되는지 추가 검증해야 합니다.

### 실 배포를 위해 추가로 필요한 것

- 다양한 배터리 및 운용 환경 데이터
- 온도, SOC, 충·방전 전류 등 실제 ESS 운용 조건 반영
- Batch 및 충전 프로토콜 변화에 강한 모델 구축
- 실제 운영 데이터 기반 외부 검증
- 지속적인 모델 성능 모니터링 및 재학습

---

# 참고문헌

- Severson, K. A. et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383–391.

---

# 팀 구성

- **서승현**
  - EDA
  - Feature Engineering
  - Feature Selection
  - Regression / Classification 모델 개발
  - Batch 2 성능 평가
  - 결과 해석 및 문서화
