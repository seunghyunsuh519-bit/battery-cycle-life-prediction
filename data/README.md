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

모델 학습 및 평가는 다음 Batch를 사용했습니다.

- **Batch 1 (2017-05-12)**: Train / Validation
- **Batch 2 (2018-02-20)**: Independent Test

원본 데이터에서 Feature Engineering을 수행한 결과는
`results/feature_dataset.csv`에 저장되어 있습니다.
