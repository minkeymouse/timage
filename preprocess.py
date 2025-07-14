import pandas as pd
import numpy as np

df_obs = pd.read_csv(
    "./data/데이터_분석과제_7_기상관측데이터_2401_2503.csv",
    encoding="utf-8",
    parse_dates=["기상관측일시"],
)

df_fcst = pd.read_csv(
    "./data/데이터_분석과제_7_기상예측데이터_2401_2503.csv",
    encoding="utf-8",
    parse_dates=["기상관측일시"],
)

obs_rename = {
    "기상관측일시":   "date",
    "습도(%)":      "humid_obs",
    "기온(degC)":   "degC_obs",
    "대기압(mmHg)": "mmHg_obs",
}

fcst_rename = {
    "기상관측일시":      "date",
    "일사량(w/m^2)":     "sunlight_fcst",
    "습도(%)":         "humid_fcst",
    "절대습도":         "abs_humid_fcst",
    "기온(degC)":      "degC_fcst",
    "대기압(hPa)":      "hPa_fcst",
}

df_obs.rename(columns=obs_rename, inplace=True)
df_fcst.rename(columns=fcst_rename, inplace=True)

df_obs = df_obs.astype({
    "humid_obs": float,
    "degC_obs":  float,
    "mmHg_obs":  float,
})
df_fcst = df_fcst.astype({
    "sunlight_fcst":    float,
    "humid_fcst":       float,
    "abs_humid_fcst":   float,
    "degC_fcst":        float,
    "hPa_fcst":         float,
})

df = pd.merge(df_obs, df_fcst, on="date", how="outer")
df = df.sort_values("date").reset_index(drop=True)

df = df.set_index("date")

full_idx = pd.date_range(
    start=df.index.min(),
    end  =df.index.max(),
    freq ="h"
)

df = (
    df
    .reindex(full_idx)
    .rename_axis("date")
    .reset_index()
)

forecast_cols = [
    "sunlight_fcst",
    "humid_fcst",
    "abs_humid_fcst",
    "degC_fcst",
    "hPa_fcst",
]

target_cols = [
    "humid_obs",
    "degC_obs",
    "mmHg_obs",
]

diff_cols = [f"{col}_diff" for col in target_cols + forecast_cols]

for col in forecast_cols:
    df[f"{col}_was_missing"] = df[col].isna().astype(str).astype("category")

solar_cutoffs = [
    (1,5),(1,20),(2,4),(2,19),(3,5),(3,20),(4,5),(4,20),
    (5,5),(5,21),(6,6),(6,21),(7,7),(7,22),(8,7),(8,23),
    (9,7),(9,23),(10,8),(10,23),(11,7),(11,22),(12,7),(12,22),
]
def solar_term_label(ts: pd.Timestamp) -> int:
    for i in range(len(solar_cutoffs)-1, -1, -1):
        m, d = solar_cutoffs[i]
        if ts >= ts.replace(month=m, day=d):
            return i + 1
    return 24

df["solar_term"] = df["date"].apply(solar_term_label).astype(str).astype("category")
df["hod"]      = df["date"].dt.hour.astype(str).astype("category")
df["dow"]      = df["date"].dt.dayofweek.astype(str).astype("category")
df["moy"]      = df["date"].dt.month.astype(str).astype("category")

seasonal_means = (
    df
    .groupby("hod", observed=False)[forecast_cols]
    .transform("mean")
)

df[forecast_cols] = df[forecast_cols].fillna(seasonal_means)

df["degC_obs_diff"]      = df["degC_obs"].diff()
df["humid_obs_diff"]     = df["humid_obs"].diff()
df["mmHg_obs_diff"]      = df["mmHg_obs"].diff()

df["degC_fcst_diff"]     = df["degC_fcst"].diff()
df["humid_fcst_diff"]    = df["humid_fcst"].diff()
df["abs_humid_fcst_diff"]= df["abs_humid_fcst"].diff()
df["hPa_fcst_diff"]      = df["hPa_fcst"].diff()
df["sunlight_fcst_diff"] = df["sunlight_fcst"].diff()

df = df.dropna(subset=target_cols + diff_cols).reset_index(drop=True)

df["group_id"] = 0
df["time_idx"] = np.arange(len(df), dtype=int)

n = len(df)

train_ratio = 0.95
test_ratio  = 0.05

n = len(df)

raw_train_end = int(n * 0.95)

df_train = df.iloc[:raw_train_end]                          
df_test   = df.iloc[raw_train_end:n]                                       

df_train.to_csv("data/train.csv", index=False)
df_test.to_csv( "data/test.csv",  index=False)
print("Preprocessing Complete!")