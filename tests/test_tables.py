import pandas as pd

from junctionmapper.pipeline import FieldResult, measure_field
from junctionmapper.tables import SCHEMAS, make_frame, write_tables
import pytest

from junctionmapper.errors import ConfigError
from tests.fixtures13 import build, make_cfg


def test_write_tables_round_trip_two_fields(tmp_path):
    results = []
    for fid in ("a", "b"):
        f, seg = build(field_id=fid)
        results.append(measure_field(f, seg, make_cfg()))
    write_tables(results, tmp_path)
    for name, cols in SCHEMAS.items():
        pq = pd.read_parquet(tmp_path / f"{name}.parquet")
        csv = pd.read_csv(tmp_path / f"{name}.csv")
        assert list(pq.columns) == cols and list(csv.columns) == cols
        assert len(pq) == len(csv) == sum(len(getattr(r, name)) for r in results)
        if name in ("membrane_patches", "membrane_fields") and not len(pq):
            continue  # measure_field does not score membranes; only `run` does
        assert set(pq.field_id) == {"a", "b"}


def test_empty_tables_keep_full_column_set(tmp_path):
    empty = FieldResult(*(make_frame(n, []) for n in SCHEMAS))
    write_tables([empty], tmp_path / "out")
    write_tables([], tmp_path / "none")
    for d in ("out", "none"):
        for name, cols in SCHEMAS.items():
            assert list(pd.read_parquet(tmp_path / d / f"{name}.parquet").columns) == cols
            assert list(pd.read_csv(tmp_path / d / f"{name}.csv").columns) == cols


def test_flags_schema_lists_all_seven_qc_fields():
    cols = set(SCHEMAS["flags"])
    assert {"border_touching", "isolated", "degenerate", "corner_methods_disagree",
            "n_corners_dp", "n_corners_tcj", "all_contacts_filtered"} <= cols


def test_nuclei_distances_table_is_written(tmp_path):
    f, seg = build()
    write_tables([measure_field(f, seg, make_cfg())], tmp_path)
    d = pd.read_parquet(tmp_path / "nuclei_distances.parquet")
    assert list(d.columns) == ["field_id", "nucleus_id", "neighbour_nucleus_id", "rank_from_nucleus",
                               "distance_px", "distance_um"]
    assert len(d) == 2 and (tmp_path / "nuclei_distances.csv").exists()


def test_duplicate_field_id_raises(tmp_path):
    rs = [measure_field(*build(field_id="same")[:2], make_cfg()) for _ in range(2)]
    with pytest.raises(ConfigError):
        write_tables(rs, tmp_path)


def test_membrane_tables_are_written_even_when_empty(tmp_path):
    empty = FieldResult(*(make_frame(n, []) for n in SCHEMAS))
    write_tables([empty], tmp_path)
    for name in ("membrane_patches", "membrane_fields"):
        assert list(pd.read_parquet(tmp_path / f"{name}.parquet").columns) == SCHEMAS[name]
    assert make_frame("membrane_patches", []).dtypes["too_dim"] == bool


def _membrane_rows(name):
    if name == "membrane_patches":
        return [dict(field_id="a", role="JM1", marker="", patch_row=0, patch_col=c, y0_px=0,
                     x0_px=c, size_px=1, region_fraction=1.0, mean_intensity=1.0,
                     ridge_intensity=1.0, background=0.0, noise=1.0, too_dim=False,
                     contrast=2.0, baseline=1.0, score=2.0, membrane_like=None) for c in (0, 1)]
    return [dict(field_id="a", role=r, marker="", n_patches=1, n_scored=1, n_too_dim=0,
                 median_score=2.0, membrane_like_fraction=float("nan")) for r in ("JM1", "JM2")]


def test_an_all_null_membrane_like_column_round_trips(tmp_path):
    frames = {n: make_frame(n, _membrane_rows(n) if n == "membrane_patches" else []) for n in SCHEMAS}
    write_tables([FieldResult(*frames.values())], tmp_path)
    pq = pd.read_parquet(tmp_path / "membrane_patches.parquet")
    assert str(pq.membrane_like.dtype) == "boolean" and pq.membrane_like.isna().all()
    csv = pd.read_csv(tmp_path / "membrane_patches.csv", keep_default_na=False)
    assert (csv.membrane_like == "").all()


def test_an_all_nan_membrane_like_fraction_round_trips(tmp_path):
    frames = {n: make_frame(n, _membrane_rows(n) if n == "membrane_fields" else []) for n in SCHEMAS}
    write_tables([FieldResult(*frames.values())], tmp_path)
    pq = pd.read_parquet(tmp_path / "membrane_fields.parquet")
    assert pq.membrane_like_fraction.dtype == "float64" and pq.membrane_like_fraction.isna().all()
    csv = pd.read_csv(tmp_path / "membrane_fields.csv", keep_default_na=False)
    assert (csv.membrane_like_fraction == "").all()
