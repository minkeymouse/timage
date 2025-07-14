import pytest
import numpy as np
import pandas as pd
import torch
from timage_forecasting.datamodule import TimeSeriesWithImage

def make_df(n_groups=2, seq_len=5, img_shape=(2,2,1)):
    """Create a toy DataFrame with time, group, numeric, categorical, and flattened image cols."""
    rows = []
    for g in range(n_groups):
        for t in range(seq_len):
            img = np.arange(np.prod(img_shape)) + g*100 + t
            rows.append({
                "time": t,
                "group": f"G{g}",
                "target": float(t + g),
                "num_feat": float(t*0.1),
                "cat_feat": f"c{t%2}",
                **{f"pix_{i}": img.flatten()[i] for i in range(img.size)}
            })
    df = pd.DataFrame(rows)
    return df

@pytest.fixture
def df():
    return make_df(n_groups=3, seq_len=4, img_shape=(2,2,1))

def test_init_requires_params(df):
    # missing time
    with pytest.raises(ValueError):
        TimeSeriesWithImage(data=df, time=None, target="target",
                             image_cols_start="pix_0", image_cols_end="pix_3", image_shape=(2,2,1))
    # missing target
    with pytest.raises(ValueError):
        TimeSeriesWithImage(data=df, time="time", target=None,
                             image_cols_start="pix_0", image_cols_end="pix_3", image_shape=(2,2,1))
    # missing image params
    with pytest.raises(ValueError):
        TimeSeriesWithImage(data=df, time="time", target="target",
                             image_cols_start=None, image_cols_end=None, image_shape=None)

def test_len_and_group_ids(df):
    ds = TimeSeriesWithImage(
        data=df, time="time", target="target", group=["group"],
        num=["num_feat"], cat=["cat_feat"], known=["num_feat"], unknown=[],
        static=None,
        image_cols_start="pix_0", image_cols_end="pix_3", image_shape=(2,2,1)
    )
    # should equal number of distinct groups
    assert len(ds) == df["group"].nunique()
    # get_metadata cols
    meta = ds.get_metadata()
    assert "y" in meta["cols"] and meta["cols"]["y"] == ["target"]
    assert "img" in meta["cols"]
    assert meta["img_size"] == (2,2,1)

def test_getitem_shapes_and_keys(df):
    ds = TimeSeriesWithImage(
        data=df, time="time", target="target", group=["group"],
        num=["num_feat"], cat=["cat_feat"], known=["num_feat"], unknown=["cat_feat"],
        static=None,
        image_cols_start="pix_0", image_cols_end="pix_3", image_shape=(2,2,1)
    )
    sample = ds[0]
    # keys
    expected_keys = {"t","y","x","img","group","st","cutoff_time","valid_mask"}
    assert expected_keys.issubset(sample.keys())
    # types
    assert isinstance(sample["t"], torch.Tensor)
    assert isinstance(sample["y"], torch.Tensor)
    assert isinstance(sample["img"], torch.Tensor)
    # lengths
    seq_len = df[df["group"]=="G0"]["time"].nunique()
    assert sample["t"].shape[0] == seq_len
    # image dims
    assert tuple(sample["img"].shape[1:]) == (2,2,1)

def test_with_future_data(df):
    # make future df with next two timesteps
    future = df.copy()
    future["time"] += 4
    ds = TimeSeriesWithImage(
        data=df, data_future=future, time="time", target="target", group=["group"],
        num=["num_feat"], cat=["cat_feat"], known=["num_feat"], unknown=["cat_feat"],
        static=None, future_image=True,
        image_cols_start="pix_0", image_cols_end="pix_3", image_shape=(2,2,1)
    )
    sample = ds[1]
    # total timeline length = past + future unique times
    total_times = np.unique(np.concatenate([
        df[df["group"]=="G1"]["time"].values,
        future[future["group"]=="G1"]["time"].values
    ]))
    assert sample["t"].shape[0] == len(total_times)
    # valid_mask has exactly past True entries
    assert sample["valid_mask"].sum().item() == df[df["group"]=="G1"].shape[0]
