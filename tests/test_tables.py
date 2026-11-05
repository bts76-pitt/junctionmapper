import pandas as pd

from junctionmapper.pipeline import FieldResult, measure_field
from junctionmapper.tables import SCHEMAS, make_frame, write_tables
from tests.test_pipeline import build, make_cfg


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
    empty = FieldResult(*(make_frame(n, []) for n in ("interfaces", "cells", "nuclei", "flags")))
    write_tables([empty], tmp_path / "out")
    write_tables([], tmp_path / "none")
    for d in ("out", "none"):
        for name, cols in SCHEMAS.items():
            assert list(pd.read_parquet(tmp_path / d / f"{name}.parquet").columns) == cols
            assert list(pd.read_csv(tmp_path / d / f"{name}.csv").columns) == cols


def test_flags_table_has_all_seven_qc_fields():
    cols = set(SCHEMAS["flags"])
    assert {"border_touching", "isolated", "degenerate", "corner_methods_disagree",
            "n_corners_dp", "n_corners_tcj", "all_contacts_filtered"} <= cols
