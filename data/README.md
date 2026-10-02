# Data

본 프로젝트는 배터리 Cycle Life 조기 예측을 위해 배터리 사이클 데이터를 사용합니다.

원본 데이터는 `.mat` 형식이며, 파일 용량 문제로 GitHub Repository에는 포함하지 않습니다.

## Data Usage

초기 100 Cycle 데이터를 사용하여 다음 정보를 추출했습니다.

- Discharge Capacity (QD)
- ΔQ(V)
- Internal Resistance (IR)
- Temperature
- Charging Time

위 정보를 기반으로 총 15개의 후보 Feature를 생성했으며,
Batch 1에서 단일 Feature별 Cross-Validation을 수행하여
최종 Regression Feature로 `deltaQ_std`를 선정했습니다.

모델 학습 및 평가는 다음 Batch를 사용했습니다.

- **Batch 1 (2017-05-12)**: Feature Selection / Train / Validation
- **Batch 2 (2018-02-20)**: Independent Test

Batch 2는 Feature 및 모델 선택 과정에 사용하지 않고,
최종 일반화 성능 평가에만 사용했습니다.

원본 데이터에서 Feature Engineering을 수행한 결과는
`results/feature_dataset.csv`에 저장되어 있습니다.
