import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from junctionmapper.config import Role
from junctionmapper.errors import JunctionMapperError, SegmentationError
from junctionmapper.io.reader import Field
from junctionmapper.segment.cellpose import SegmentationParams, segment_field

# Each plane is filled with a distinct constant so the stub can tell which one it got.
FILL = {Role.JM1: 1.0, Role.JM2: 2.0, Role.NUCLEUS: 3.0, Role.SEGMENT: 4.0}
NAME_OF_FILL = {1.0: "JM1", 2.0: "JM2", 3.0: "NUCLEUS", 4.0: "SEGMENT"}


def synthetic_field(roles=(Role.SEGMENT, Role.NUCLEUS), shape=(16, 20)) -> Field:
    return Field(
        planes={r: np.full(shape, FILL[r], dtype=np.float32) for r in roles},
        pixel_size_um=0.325,
        markers={r: "m" for r in roles},
        source=Path("x.tif"),
        field_id="x_s0_t0_z0",
    )


def stub_cellpose(monkeypatch, *, version="4.0.1", diams=None, device=None, mutate=False):
    """Install fake cellpose (+ cellpose.models); return a record of calls."""
    rec = {"init": [], "eval": []}

    class FakeModel:
        def __init__(self, **kwargs):
            rec["init"].append(kwargs)
            if device is not None:
                self.device = device

        def eval(self, plane, **kwargs):
            name = NAME_OF_FILL[float(plane.flat[0])]
            rec["eval"].append({"plane": name, "shape": plane.shape, "x": plane.copy(), **kwargs})
            if mutate:
                # Real Cellpose percentile-normalises a float32 input IN PLACE.
                plane -= plane.min()
                top = plane.max()
                plane /= top if top else 1.0
            if name == "SEGMENT":
                masks = np.zeros(plane.shape, dtype=np.uint16)
                masks[2:6, 2:6] = 1
                masks[8:12, 8:12] = 2
            else:
                masks = np.zeros(plane.shape, dtype=np.int32)
                masks[1:3, 14:19] = 5
                masks[13:15, 1:9] = 7
            if diams is None:
                return masks, [], []
            return masks, [], [], diams

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = FakeModel
    pkg = types.ModuleType("cellpose")
    pkg.models = models
    pkg.__version__ = version
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return rec


def stub_torch(monkeypatch, *, cuda: bool):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.5.0"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)


def record_eval_calls(monkeypatch):
    return stub_cellpose(monkeypatch)["eval"]


def stub_no_gpu(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    stub_torch(monkeypatch, cuda=False)
    return rec


def test_segment_field_calls_cellpose_on_segment_and_nucleus_planes(monkeypatch):
    seen = record_eval_calls(monkeypatch)
    segment_field(synthetic_field(), SegmentationParams())
    assert [c["plane"] for c in seen] == ["SEGMENT", "NUCLEUS"]


def test_params_used_records_everything_needed_to_reproduce(monkeypatch):
    stub_cellpose(monkeypatch)
    used = segment_field(synthetic_field(), SegmentationParams(diameter=30)).params_used
    assert used["model"] == "cpsam" and used["diameter"] == 30
    assert "cellpose_version" in used


def test_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    rec = stub_no_gpu(monkeypatch)
    seg = segment_field(synthetic_field(), SegmentationParams(gpu=True))
    assert seg.params_used["gpu"] is False
    assert seg.params_used["gpu_requested"] is True
    assert rec["init"][0]["gpu"] is False  # fallback still segments
    assert seg.cells.max() == 2


def test_uses_gpu_when_available(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    stub_torch(monkeypatch, cuda=True)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=True)).params_used
    assert used["gpu"] is True and rec["init"][0]["gpu"] is True


def test_gpu_false_requested_is_honoured(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    stub_torch(monkeypatch, cuda=True)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=False)).params_used
    assert used["gpu"] is False and rec["init"][0]["gpu"] is False


def test_thresholds_and_model_passed_to_cellpose(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    p = SegmentationParams(diameter=25.0, flow_threshold=0.2, cellprob_threshold=-1.0)
    segment_field(synthetic_field(), p)
    assert rec["init"][0]["pretrained_model"] == "cpsam"
    for call in rec["eval"]:
        assert call["diameter"] == 25.0
        assert call["flow_threshold"] == 0.2
        assert call["cellprob_threshold"] == -1.0


def test_cells_are_segment_labels_and_nuclei_are_nucleus_labels(monkeypatch):
    stub_cellpose(monkeypatch)
    seg = segment_field(synthetic_field(shape=(16, 20)), SegmentationParams())
    exp_cells = np.zeros((16, 20), dtype=np.uint16)
    exp_cells[2:6, 2:6] = 1
    exp_cells[8:12, 8:12] = 2
    exp_nuc = np.zeros((16, 20), dtype=np.int32)
    exp_nuc[1:3, 14:19] = 5
    exp_nuc[13:15, 1:9] = 7
    np.testing.assert_array_equal(seg.cells, exp_cells)
    np.testing.assert_array_equal(seg.nuclei, exp_nuc)
    assert seg.cells.dtype == np.uint16 and seg.nuclei.dtype == np.int32


def test_diameter_none_without_report_is_recorded_as_none(monkeypatch):
    stub_cellpose(monkeypatch)
    used = segment_field(synthetic_field(), SegmentationParams()).params_used
    assert used["diameter"] is None and used["diameter_used"] is None


def test_diameter_none_records_reported_estimate(monkeypatch):
    stub_cellpose(monkeypatch, diams=31.5)
    used = segment_field(synthetic_field(), SegmentationParams()).params_used
    assert used["diameter"] is None
    assert used["diameter_used"] == {"SEGMENT": 31.5, "NUCLEUS": 31.5}


def test_missing_nucleus_gives_zero_array_and_is_skipped(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    seg = segment_field(synthetic_field(roles=(Role.SEGMENT,)), SegmentationParams())
    assert [c["plane"] for c in rec["eval"]] == ["SEGMENT"]
    assert seg.nuclei.shape == seg.cells.shape and not seg.nuclei.any()
    assert seg.nuclei.dtype == seg.cells.dtype == np.uint16
    assert seg.params_used["roles_segmented"] == ["SEGMENT"]


def test_measured_planes_are_never_segmented(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    segment_field(synthetic_field(roles=(Role.SEGMENT, Role.JM1, Role.JM2, Role.NUCLEUS)), SegmentationParams())
    assert [c["plane"] for c in rec["eval"]] == ["SEGMENT", "NUCLEUS"]


def test_nucleus_only_field_raises_segmentation_error(monkeypatch):
    stub_cellpose(monkeypatch)
    with pytest.raises(SegmentationError, match="segment"):
        segment_field(synthetic_field(roles=(Role.NUCLEUS,)), SegmentationParams())


def test_params_used_is_json_serialisable(monkeypatch):
    stub_cellpose(monkeypatch, diams=np.float32(30.0))
    stub_torch(monkeypatch, cuda=False)
    used = segment_field(synthetic_field(), SegmentationParams()).params_used
    json.dumps(used)


def test_absent_cellpose_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)  # forces ImportError
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    with pytest.raises(SegmentationError, match=r"junctionmapper\[segmentation\]") as ei:
        segment_field(synthetic_field(), SegmentationParams())
    assert isinstance(ei.value, JunctionMapperError)
    assert not isinstance(ei.value, ModuleNotFoundError)


def test_module_imports_without_cellpose():
    import importlib

    import junctionmapper.segment.cellpose as mod

    sys.modules.pop("cellpose", None)
    importlib.reload(mod)
    assert "cellpose" not in sys.modules  # import must be lazy, not at module scope


def test_non_default_model_reaches_cellpose_and_params_used(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    used = segment_field(synthetic_field(), SegmentationParams(model="my_custom")).params_used
    assert rec["init"][0]["pretrained_model"] == "my_custom"
    assert used["model"] == "my_custom"


def test_device_and_versions_recorded_nonempty(monkeypatch):
    stub_cellpose(monkeypatch, version="4.0.1")
    stub_torch(monkeypatch, cuda=False)
    used = segment_field(synthetic_field(), SegmentationParams()).params_used
    assert used["device"] == "cpu" and used["device_source"] == "wrapper_selection"
    assert used["torch_version"] == "2.5.0"
    assert used["cellpose_version"]
    assert used["field_id"] == "x_s0_t0_z0"


def test_device_records_selected_backend_when_gpu_used(monkeypatch):
    stub_cellpose(monkeypatch)
    stub_torch(monkeypatch, cuda=True)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=True)).params_used
    assert used["device"] == "cuda" and used["device_source"] == "wrapper_selection"


def test_cpu_never_reported_as_accelerator_when_gpu_not_requested(monkeypatch):
    stub_cellpose(monkeypatch)
    stub_torch(monkeypatch, cuda=True)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=False)).params_used
    assert used["device"] == "cpu"


def test_model_reported_device_is_recorded_and_overrides_gpu_choice(monkeypatch):
    stub_cellpose(monkeypatch, device=types.SimpleNamespace(type="cpu"))
    stub_torch(monkeypatch, cuda=True)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=True)).params_used
    assert used["gpu"] is False  # model fell back despite torch saying cuda
    assert used["device_source"] == "model"
    assert "cpu" in used["device"]


def test_model_reported_cuda_device_sets_gpu_true(monkeypatch):
    stub_cellpose(monkeypatch, device=types.SimpleNamespace(type="cuda"))
    stub_torch(monkeypatch, cuda=False)
    used = segment_field(synthetic_field(), SegmentationParams(gpu=True)).params_used
    assert used["gpu"] is True and used["device_source"] == "model"


def test_segment_field_does_not_mutate_the_callers_planes(monkeypatch):
    """Cellpose normalises a float32 input in place; the field's planes must survive.

    Regression: ``run`` segments and measures the SAME Field in one process, so a plane
    mutated by Cellpose silently became the measurement input. Thresholds are calibrated
    on the raw planes beforehand, so every marker metric collapsed to zero while the
    geometry columns stayed plausible -- a silent wrong answer, not a crash.
    """
    stub_cellpose(monkeypatch, mutate=True)
    field = synthetic_field()
    before = {r: p.copy() for r, p in field.planes.items()}
    segment_field(field, SegmentationParams(gpu=False))
    for role, plane in field.planes.items():
        np.testing.assert_array_equal(
            plane, before[role], err_msg=f"{role} plane was mutated by segmentation"
        )


# ---- cells come from the segmentation channel, never from a measured one ----------------

def test_cells_are_segmented_from_the_segment_plane(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    seg = segment_field(synthetic_field(roles=(Role.SEGMENT, Role.JM1, Role.JM2, Role.NUCLEUS)),
                        SegmentationParams())
    assert [c["plane"] for c in rec["eval"]] == ["SEGMENT", "NUCLEUS"]
    assert seg.params_used["roles_segmented"] == ["SEGMENT", "NUCLEUS"]
    assert seg.cells.max() == 2  # the cell masks the stub returns for SEGMENT


def test_segmenting_without_a_segment_plane_is_refused(monkeypatch):
    rec = stub_cellpose(monkeypatch)
    with pytest.raises(SegmentationError, match="segment"):
        segment_field(synthetic_field(roles=(Role.JM1, Role.NUCLEUS)), SegmentationParams())
    assert rec["eval"] == []  # refused before any Cellpose call


# ---- with_nucleus: the nucleus plane as Cellpose's second channel for cells -------------

def test_cells_are_segmented_on_the_segment_plane_alone_by_default(monkeypatch):
    seen = record_eval_calls(monkeypatch)
    segment_field(synthetic_field(), SegmentationParams())
    cell = seen[0]
    assert cell["shape"] == (16, 20)
    assert "channel_axis" not in cell


def test_with_nucleus_gives_cellpose_segment_then_nucleus_channels(monkeypatch):
    """A cell boundary drawn from the membrane plane alone can cut straight through a
    nucleus; Cellpose-SAM avoids that when it also sees the nuclei."""
    seen = record_eval_calls(monkeypatch)
    segment_field(synthetic_field(), SegmentationParams(with_nucleus=True))
    cell = seen[0]
    assert cell["shape"] == (16, 20, 2)
    assert cell["channel_axis"] == -1
    assert np.all(cell["x"][..., 0] == FILL[Role.SEGMENT])
    assert np.all(cell["x"][..., 1] == FILL[Role.NUCLEUS])


def test_with_nucleus_still_segments_nuclei_on_the_nucleus_plane_alone(monkeypatch):
    seen = record_eval_calls(monkeypatch)
    segment_field(synthetic_field(), SegmentationParams(with_nucleus=True))
    nuc = seen[1]
    assert nuc["plane"] == "NUCLEUS" and nuc["shape"] == (16, 20)
    assert "channel_axis" not in nuc


def test_with_nucleus_without_a_nucleus_plane_is_refused(monkeypatch):
    record_eval_calls(monkeypatch)
    with pytest.raises(SegmentationError, match="nucleus"):
        segment_field(synthetic_field(roles=(Role.SEGMENT,)), SegmentationParams(with_nucleus=True))


def test_params_used_records_which_planes_the_cells_were_drawn_from(monkeypatch):
    stub_no_gpu(monkeypatch)
    plain = segment_field(synthetic_field(), SegmentationParams()).params_used
    both = segment_field(synthetic_field(), SegmentationParams(with_nucleus=True)).params_used
    assert plain["with_nucleus"] is False and plain["cell_input"] == ["SEGMENT"]
    assert both["with_nucleus"] is True and both["cell_input"] == ["SEGMENT", "NUCLEUS"]
