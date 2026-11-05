import pytest
from pydantic import ValidationError

from junctionmapper.config import ChannelSpec, RunConfig, Role


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


def test_dilation_um_requires_pixel_size():
    with pytest.raises(ValidationError):
        RunConfig(channels=_minimal_channels(), pixel_size_um=None, dilation_um=1.5)
