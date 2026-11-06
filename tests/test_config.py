import pytest
from pydantic import ValidationError

from junctionmapper.config import (
    DEFAULT_NEIGHBOUR_RADIUS_PX, DEFAULT_TCJ_MIN_RUN, ChannelSpec, RunConfig, Role,
)


def test_role_lookup_returns_marker_identity():
    cfg = RunConfig(
        channels=[
            ChannelSpec(role=Role.JM1, channel="NaK", marker="Na/K-ATPase"),
            ChannelSpec(role=Role.NUCLEUS, channel="DAPI", marker="DAPI"),
        ],
        pixel_size_um=0.325,
    )
    assert cfg.marker_for(Role.JM1) == "Na/K-ATPase"
    assert cfg.role_for(Role.NUCLEUS).channel == "DAPI"


def test_duplicate_role_rejected():
    with pytest.raises(ValidationError):
        RunConfig(
            channels=[
                ChannelSpec(role=Role.JM1, channel=0, marker="CDH1"),
                ChannelSpec(role=Role.JM1, channel=1, marker="NaK"),
                ChannelSpec(role=Role.NUCLEUS, channel=2, marker="DAPI"),
            ],
            pixel_size_um=0.325,
        )


def test_jm1_and_nucleus_are_required():
    with pytest.raises(ValidationError):
        RunConfig(
            channels=[ChannelSpec(role=Role.JM2, channel=0, marker="p120")],
            pixel_size_um=0.325,
        )


def test_jm2_is_optional():
    cfg = RunConfig(
        channels=[
            ChannelSpec(role=Role.JM1, channel=0, marker="m1"),
            ChannelSpec(role=Role.NUCLEUS, channel=1, marker="DAPI"),
        ],
        pixel_size_um=0.325,
    )
    with pytest.raises(KeyError):
        cfg.role_for(Role.JM2)


def _minimal_channels():
    return [
        ChannelSpec(role=Role.JM1, channel=0, marker="m1"),
        ChannelSpec(role=Role.NUCLEUS, channel=1, marker="DAPI"),
    ]


def test_defaults():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=None)
    assert cfg.dilation_px == 2
    assert cfg.num_intra_cells == 5
    assert cfg.dilation_um is None
    assert cfg.all_pairs is False
    assert cfg.threshold is None
    assert cfg.resolved_dilation_px() == 2


def test_dilation_um_converts_to_px_using_pixel_size():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=0.5, dilation_um=1.5)
    assert cfg.resolved_dilation_px() == 3


def test_dilation_um_overrides_dilation_px():
    cfg = RunConfig(
        channels=_minimal_channels(), pixel_size_um=0.5, dilation_px=9, dilation_um=1.5
    )
    assert cfg.resolved_dilation_px() == 3


def test_resolved_dilation_has_minimum_of_one():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=1.0, dilation_um=0.1)
    assert cfg.resolved_dilation_px() == 1


def test_config_with_dilation_um_and_no_pixel_size_is_valid():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=None, dilation_um=1.5)
    assert cfg.resolved_dilation_px(pixel_size_um=0.5) == 3


def test_pixel_size_argument_overrides_config_fallback():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=0.5, dilation_um=1.5)
    assert cfg.resolved_dilation_px() == 3
    assert cfg.resolved_dilation_px(pixel_size_um=0.25) == 6


def test_dilation_um_without_any_pixel_size_raises_at_call_time():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=None, dilation_um=1.5)
    with pytest.raises(ValueError):
        cfg.resolved_dilation_px()


def test_reproducibility_parameters_default_and_validate():
    cfg = RunConfig(channels=_minimal_channels(), pixel_size_um=0.3)
    assert (cfg.epsilon_px, cfg.tcj_min_run, cfg.neighbour_radius_px) == (2.0, 3, 2)
    for bad in ({"epsilon_px": 0}, {"epsilon_px": float("nan")}, {"epsilon_px": True},
                {"tcj_min_run": 0}, {"tcj_min_run": True}, {"tcj_min_run": 2.5},
                {"neighbour_radius_px": 0}, {"neighbour_radius_px": False}):
        with pytest.raises(ValidationError):
            RunConfig(channels=_minimal_channels(), pixel_size_um=0.3, **bad)


def _cfg(**kw):
    return RunConfig(
        channels=[ChannelSpec(role=Role.JM1, channel=0, marker="CDH1"),
                  ChannelSpec(role=Role.NUCLEUS, channel=1, marker="DAPI")],
        pixel_size_um=0.325, **kw)


def test_unknown_key_is_rejected_not_silently_dropped():
    """M6: a misspelled key used to be dropped, so the run went ahead at a default the
    operator believed they had changed."""
    with pytest.raises(ValidationError):
        _cfg(epsilon=3.0)                       # it is epsilon_px
    with pytest.raises(ValidationError):
        _cfg(tcj_minrun=5)


def test_a_nucleus_threshold_is_rejected_because_nothing_reads_it():
    with pytest.raises(ValidationError):
        _cfg(threshold={Role.NUCLEUS: 10.0})
    assert _cfg(threshold={Role.JM1: 10.0}).threshold == {Role.JM1: 10.0}


def test_geometry_defaults_have_a_single_definition():
    """M5: the literals used to be repeated across config, geometry and qc."""
    from junctionmapper.geometry import corners_tcj, interfaces
    from junctionmapper import qc
    cfg = _cfg()
    assert cfg.tcj_min_run == DEFAULT_TCJ_MIN_RUN
    assert cfg.neighbour_radius_px == DEFAULT_NEIGHBOUR_RADIUS_PX
    assert interfaces.TCJ_MIN_RUN is DEFAULT_TCJ_MIN_RUN
    assert qc._NEIGHBOUR_RADIUS is DEFAULT_NEIGHBOUR_RADIUS_PX
    import inspect
    assert (inspect.signature(corners_tcj.tcj_corners).parameters["min_run"].default
            is DEFAULT_TCJ_MIN_RUN)
    assert (inspect.signature(corners_tcj.neighbour_at).parameters["search_radius"].default
            is DEFAULT_NEIGHBOUR_RADIUS_PX)
    assert (inspect.signature(interfaces.extract_interfaces).parameters["search_radius"].default
            is DEFAULT_NEIGHBOUR_RADIUS_PX)
    assert (inspect.signature(qc.assess_cell).parameters["min_run"].default
            is DEFAULT_TCJ_MIN_RUN)
