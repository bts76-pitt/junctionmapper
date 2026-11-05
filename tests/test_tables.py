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
    assert list(d.columns) == ["field_id", "nucleus_id", "neighbour_nucleus_id", "rank",
                               "distance_px", "distance_um"]
    assert len(d) == 2 and (tmp_path / "nuclei_distances.csv").exists()


def test_duplicate_field_id_raises(tmp_path):
    rs = [measure_field(*build(field_id="same")[:2], make_cfg()) for _ in range(2)]
    with pytest.raises(ConfigError):
        write_tables(rs, tmp_path)
