import ast
import inspect
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from junctionmapper import config as config_mod
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


def _module_tree(module) -> ast.Module:
    return ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))


def _assignments(body) -> dict:
    """Every ``NAME = <expr>`` / ``NAME: T = <expr>`` in ``body``, as written in the source."""
    out = {}
    for node in body:
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and getattr(node, "value", None) is not None:
                out[t.id] = node.value
    return out


def _class_assignments(module, cls_name: str) -> dict:
    for node in _module_tree(module).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return _assignments(node.body)
    raise AssertionError(f"{module.__name__} has no class {cls_name}")


def _written_default(func, param: str) -> ast.expr:
    """The default of ``param`` as it is WRITTEN in ``func``'s source, not as evaluated."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    a = tree.args
    positional = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
    keyword = [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    for arg, node in positional + keyword:
        if arg.arg == param:
            return node
    raise AssertionError(f"{func.__qualname__} has no defaulted parameter {param!r}")


def _names_the_constant(module, node: ast.expr, want: str, chain=()) -> list:
    """Follow module-level aliases from ``node`` to ``want``, structurally.

    Raises if any link is a literal rather than a name: that is the mutant this is here for.
    """
    assert not isinstance(node, ast.Constant), (
        f"a literal {node.value!r} is written here instead of {want}; the point of a single "
        f"definition is that no other site re-states the value")
    assert isinstance(node, (ast.Name, ast.Attribute)), ast.dump(node)
    name = node.id if isinstance(node, ast.Name) else node.attr
    if name == want:
        return [*chain, name]
    assert name not in chain, f"alias cycle {chain}"
    alias = _assignments(_module_tree(module).body).get(name)
    assert alias is not None, f"{module.__name__}.{name} is not a module-level alias of {want}"
    return _names_the_constant(module, alias, want, (*chain, name))


def test_geometry_defaults_have_a_single_definition():
    """M5: the literals used to be repeated across config, geometry and qc.

    Checked on the source of each call site, not by identity on the evaluated default: CPython
    interns small ints, so ``inspect.signature(...).default is DEFAULT_TCJ_MIN_RUN`` holds for
    a re-hardcoded ``3`` as well and could not tell a shared constant from a copied literal.
    Here every call site's default must be WRITTEN as a name, and that name must reach the
    constant through module-level aliases that are themselves names.
    """
    from junctionmapper.geometry import corners_tcj, interfaces
    from junctionmapper import qc
    cfg = _cfg()
    assert cfg.tcj_min_run == DEFAULT_TCJ_MIN_RUN
    assert cfg.neighbour_radius_px == DEFAULT_NEIGHBOUR_RADIUS_PX
    assert interfaces.TCJ_MIN_RUN == DEFAULT_TCJ_MIN_RUN
    assert qc._NEIGHBOUR_RADIUS == DEFAULT_NEIGHBOUR_RADIUS_PX
    for module, func, param, want in [
        (corners_tcj, corners_tcj.tcj_corners, "min_run", "DEFAULT_TCJ_MIN_RUN"),
        (qc, qc.assess_cell, "min_run", "DEFAULT_TCJ_MIN_RUN"),
        (interfaces, interfaces.extract_interfaces, "min_run", "DEFAULT_TCJ_MIN_RUN"),
        (corners_tcj, corners_tcj.neighbour_at, "search_radius", "DEFAULT_NEIGHBOUR_RADIUS_PX"),
        (qc, qc.assess_cell, "search_radius", "DEFAULT_NEIGHBOUR_RADIUS_PX"),
        (interfaces, interfaces.extract_interfaces, "search_radius",
         "DEFAULT_NEIGHBOUR_RADIUS_PX"),
    ]:
        _names_the_constant(module, _written_default(func, param), want)
    # and the two defaults RunConfig itself declares
    fields = _class_assignments(config_mod, "RunConfig")
    for name, want in (("tcj_min_run", "DEFAULT_TCJ_MIN_RUN"),
                       ("neighbour_radius_px", "DEFAULT_NEIGHBOUR_RADIUS_PX")):
        assert name in fields, f"RunConfig.{name} has no written default"
        _names_the_constant(config_mod, fields[name], want)


def test_the_single_definition_check_rejects_a_re_hardcoded_literal():
    """The guard above is only worth having if it fails on the mutant it is named for."""
    src = "def f(a, min_run: int = 3):\n    return a\n"
    node = ast.parse(src).body[0].args.defaults[0]
    with pytest.raises(AssertionError, match="a literal 3 is written here"):
        _names_the_constant(config_mod, node, "DEFAULT_TCJ_MIN_RUN")
