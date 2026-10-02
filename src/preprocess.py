import os
import glob

import mat73
import scipy.io
import numpy as np
import pandas as pd


def load_battery_data(data_dir):
    """
    지정한 폴더에서 batchdata .mat 파일을 불러온다.

    Parameters
    ----------
    data_dir : str
        원본 .mat 파일이 저장된 폴더 경로

    Returns
    -------
    dict
        {filename: loaded_mat_data}
    """
    mat_files = sorted(
        glob.glob(os.path.join(data_dir, "*batchdata*.mat"))
    )

    if not mat_files:
        raise FileNotFoundError(
            f"batchdata .mat 파일을 찾을 수 없습니다: {data_dir}"
        )

    mats = {}

    for path in mat_files:
        filename = os.path.basename(path)
        print(f"로딩 중... ({filename})")

        try:
            data = mat73.loadmat(path)
        except Exception:
            data = scipy.io.loadmat(
                path,
                simplify_cells=True
            )

        mats[filename] = data

    print(f"\n총 {len(mats)}개 .mat 파일 로딩 완료")

    return mats


def _safe_val(arr, cycle_idx, is_mean=True):
    """
    None, 빈 배열 등 예외를 처리하면서 cycle 값을 안전하게 추출한다.
    """
    if arr is None or cycle_idx >= len(arr):
        return 0.0

    val = arr[cycle_idx]

    if val is None:
        return 0.0

    try:
        if (
            is_mean
            and hasattr(val, "__len__")
            and not isinstance(val, (str, bytes))
            and len(val) > 0
        ):
            return float(np.mean(val))

        return float(val)

    except Exception:
        return 0.0


def build_cycle_dataframe(mats):
    rows = []

    for filename, mat_data in mats.items():
        batch_name = filename.split("_")[0]

        main_key = (
            "batch"
            if "batch" in mat_data
            else [k for k in mat_data.keys() if not k.startswith("__")][0]
        )

        batch_struct = mat_data[main_key]

        cycles = batch_struct["cycles"]
        summaries = batch_struct["summary"]
        policies = batch_struct["policy"]
        official_cycle_life = batch_struct["cycle_life"]

        for idx, cell in enumerate(cycles):

            v = cell.get("V")

            if v is None:
                continue

            qd = cell.get("Qd")
            qc = cell.get("Qc")

            summary = summaries[idx]

            ir = summary.get("IR")
            tavg = summary.get("Tavg")
            tmax = summary.get("Tmax")
            chargetime = summary.get("chargetime")

            # 실제 기록된 Cycle 수
            n_cycles = len(v)

            # 공식 Cycle Life
            try:
                target_life = float(
                    np.asarray(
                        official_cycle_life[idx]
                    ).squeeze()
                )
            except (TypeError, ValueError):
                target_life = np.nan

            for cycle_idx in range(n_cycles):

                rows.append({
                    "cell_id": f"{batch_name}_cell_{idx}",
                    "batch": batch_name,
                    "cycle": cycle_idx + 1,
                    "cycle_life": target_life,

                    "charging_policy": (
                        str(policies[idx])
                        if policies is not None
                        else "Unknown"
                    ),

                    "QD": _safe_val(
                        qd, cycle_idx, True
                    ),

                    "QC": _safe_val(
                        qc, cycle_idx, True
                    ),

                    "IR": _safe_val(
                        ir, cycle_idx, False
                    ),

                    "Tavg": _safe_val(
                        tavg, cycle_idx, False
                    ),

                    "Tmax": _safe_val(
                        tmax, cycle_idx, False
                    ),

                    "chargetime": _safe_val(
                        chargetime, cycle_idx, False
                    )
                })

    df = pd.DataFrame(rows)

    print(
        f"\n[통합 완료] 총 데이터 행 수: {len(df):,}, "
        f"총 셀 개수: {df['cell_id'].nunique()}"
    )

    print(
        "포함된 배치 목록:",
        df["batch"].unique()
    )

    return df