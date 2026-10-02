import numpy as np
import pandas as pd


def calculate_qd_features(early_df):
    """
    초기 Cycle의 방전 용량(QD) Feature 생성
    """

    def _calculate(group):
        valid = group[group["QD"] > 0].copy()

        if len(valid) < 2:
            return pd.Series({
                "mean_QD": np.nan,
                "std_QD": np.nan,
                "QD_slope": np.nan
            })

        return pd.Series({
            "mean_QD": valid["QD"].mean(),
            "std_QD": valid["QD"].std(),
            "QD_slope": np.polyfit(
                valid["cycle"],
                valid["QD"],
                1
            )[0]
        })

    return (
        early_df
        .groupby("cell_id")
        .apply(_calculate)
        .reset_index()
    )


def calculate_operational_features(early_df):
    """
    초기 Cycle의 내부저항, 온도, 충전시간 Feature 생성
    """

    def _calculate(group):
        features = {}

        # Internal Resistance
        valid_ir = group[group["IR"] > 0]

        if len(valid_ir) >= 2:
            features["mean_IR"] = valid_ir["IR"].mean()
            features["IR_slope"] = np.polyfit(
                valid_ir["cycle"],
                valid_ir["IR"],
                1
            )[0]
        else:
            features["mean_IR"] = np.nan
            features["IR_slope"] = np.nan

        # Temperature
        valid_tavg = group[group["Tavg"] > 0]["Tavg"]
        valid_tmax = group[group["Tmax"] > 0]["Tmax"]

        features["mean_Tavg"] = (
            valid_tavg.mean()
            if len(valid_tavg) > 0
            else np.nan
        )

        features["mean_Tmax"] = (
            valid_tmax.mean()
            if len(valid_tmax) > 0
            else np.nan
        )

        features["max_Tmax"] = (
            valid_tmax.max()
            if len(valid_tmax) > 0
            else np.nan
        )

        # Charging Time
        valid_ct = group[
            group["chargetime"] > 0
        ]["chargetime"]

        features["mean_chargetime"] = (
            valid_ct.mean()
            if len(valid_ct) > 0
            else np.nan
        )

        return pd.Series(features)

    return (
        early_df
        .groupby("cell_id")
        .apply(_calculate)
        .reset_index()
    )


def merge_features(
    df,
    qd_features,
    delta_qv_features,
    operational_features
):
    """
    Cell 정보와 생성된 Feature들을 하나의 Dataset으로 병합
    """

    cell_info = (
        df[
            [
                "cell_id",
                "batch",
                "cycle_life",
                "charging_policy"
            ]
        ]
        .drop_duplicates(subset="cell_id")
        .reset_index(drop=True)
    )

    feature_df = (
        cell_info
        .merge(
            qd_features,
            on="cell_id",
            how="left"
        )
        .merge(
            delta_qv_features,
            on="cell_id",
            how="left"
        )
        .merge(
            operational_features,
            on="cell_id",
            how="left"
        )
    )

    return feature_df