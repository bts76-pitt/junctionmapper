"""Outline-free membrane score (``measure/membrane.py``).

Fixture: pixel size 0.1 um, 400 x 400 = 2 x 2 patches of 20 um. One nucleus (disk of
radius 10 px, value 200) at the centre of each 50 px cell; the cell region grown 6 um
from them covers the whole field. Cell borders are lines centred on multiples of 50 px.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from junctionmapper.config import MembraneScoreConfig, Role
from junctionmapper.io.reader import Field
from junctionmapper.measure.membrane import (_filter_geometry, cell_region, floor_and_noise,
                                             score_field)

N = 400          # field side, px
CELL = 50        # cell pitch, px
LINE = 5         # border line width, px
CALL = MembraneScoreConfig(cutoff=1.5)     # the default makes no membrane_like call


def _yx():
    return np.mgrid[0:N, 0:N]


def _borders():
    """True on the 5 px wide grid lines centred on every multiple of 50 px."""
    y, x = _yx()
    return ((y + LINE // 2) % CELL < LINE) | ((x + LINE // 2) % CELL < LINE)


def _noise(seed=0):
    return np.random.default_rng(seed).normal(0, 2, (N, N))


def nuclei():
    y, x = _yx()
    dy = (y % CELL) - CELL // 2
    dx = (x % CELL) - CELL // 2
    return np.where(dy ** 2 + dx ** 2 <= 10 ** 2, 200.0, 0.0).astype(np.float32)


def lines():
    """Grid lines 100 on 10, plus Gaussian noise sd 2."""
    return (np.where(_borders(), 100.0, 10.0) + _noise()).astype(np.float32)


def noise():
    """50 plus Gaussian noise sd 2: no structure at all."""
    return (50.0 + _noise()).astype(np.float32)


def noise_on_floor():
    """``noise()`` with 2% of pixels, at random, at 0: the same structureless signal, but
    the field now has a dark floor for its 1st percentile to find, as real fields do."""
    img = noise().ravel()
    img[np.random.default_rng(1).choice(N * N, N * N // 50, replace=False)] = 0.0
    return img.reshape(N, N)


def filled():
    """Each cell interior 80, borders 10, smoothed sigma 2."""
    return ndi.gaussian_filter(np.where(_borders(), 10.0, 80.0), 2).astype(np.float32)


def sparse_ones():
    """Zeros with 0.1% ones: a channel at quantisation level."""
    img = np.zeros(N * N, np.float32)
    img[np.random.default_rng(0).choice(N * N, N * N // 1000, replace=False)] = 1.0
    return img.reshape(N, N)


def field_with(jm1, jm2=None, segment=None, nucleus=None, px=0.1) -> Field:
    planes = {Role.JM1: np.asarray(jm1, np.float32),
              Role.NUCLEUS: nuclei() if nucleus is None else np.asarray(nucleus, np.float32)}
    markers = {Role.JM1: "E-cadherin", Role.NUCLEUS: "DAPI"}
    if jm2 is not None:
        planes[Role.JM2], markers[Role.JM2] = np.asarray(jm2, np.float32), "p120"
    if segment is not None:
        planes[Role.SEGMENT], markers[Role.SEGMENT] = np.asarray(segment, np.float32), "ZO-1"
    return Field(planes=planes, pixel_size_um=px, markers=markers,
                 source=Path("x.tif"), field_id="f0")


def small_field() -> Field:
    """40 x 40 at 0.1 um: smaller than one 200 px patch."""
    f = field_with(jm1=lines())
    f.planes = {r: p[:40, :40].copy() for r, p in f.planes.items()}
    return f


# --- the score -----------------------------------------------------------------------

def test_thin_lines_are_membrane_like():
    patches, fields = score_field(field_with(jm1=lines()), CALL)
    s = patches[patches.role == "JM1"]
    assert len(s) == 4 and not s.too_dim.any() and (s.score >= 1.5).all() and s.membrane_like.all()
    assert fields.set_index("role").loc["JM1", "membrane_like_fraction"] == pytest.approx(1.0)


def test_uniform_noise_scores_near_one():
    s = score_field(field_with(jm1=noise_on_floor()), CALL)[0]
    assert s.score.between(0.95, 1.05).all() and not s.membrane_like.any()     # measured 0.999 to 1.001


def test_uniform_noise_with_no_floor_is_too_dim():
    # Pure noise puts about 4 x noise on the ridge pixels above the 1st percentile, under
    # the 5 x noise cut: nothing stands out from the floor, so too dim, not diffuse.
    s = score_field(field_with(jm1=noise()), CALL)[0]
    assert s.too_dim.all() and s.score.isna().all()


def test_filled_cells_are_not_membrane_like():
    s = score_field(field_with(jm1=filled()), CALL)[0]
    assert (s.score < 1.1).all()                                            # measured 1.009 to 1.015


def test_a_blank_channel_is_too_dim_not_diffuse():
    # membrane_like is null, not False, where there is no score: a mean over the column
    # must not read a blank channel as 0% membrane-like (diffuse).
    s = score_field(field_with(jm1=sparse_ones()), CALL)[0]
    assert s.too_dim.all() and s.score.isna().all() and s.membrane_like.isna().all()
    assert str(s.membrane_like.dtype) == "boolean"


def test_bright_lines_on_a_mostly_exact_zero_field_are_scored():
    # 62% of the field (every column left of 248) is exactly 0, so a whole-field MAD of
    # neighbour differences would be 0. The lines are bright: they must be scored, not too dim.
    img = lines()
    img[:, :248] = 0.0
    p, _ = score_field(field_with(jm1=img), CALL)
    right, left = p[p.x0_px == 200], p[p.x0_px == 0]
    assert (p.noise > 0).all()
    assert not right.too_dim.any() and right.membrane_like.all()
    assert left.too_dim.all()                     # all floor: genuinely no signal


def sparse_lines():
    """Grid lines 3 px wide every 100 px, 30 on 10, noise sd 2: about 6% of pixels are
    membrane, so the region mean sits under 3 x noise above the floor (as dim 8-bit fields
    do) while the ridge pixels stand well clear of it."""
    y, x = _yx()
    on = ((y + 1) % 100 < 3) | ((x + 1) % 100 < 3)
    return (np.where(on, 30.0, 10.0) + _noise()).astype(np.float32)


def test_thin_lines_with_a_low_region_mean_are_scored_not_too_dim():
    p = score_field(field_with(jm1=sparse_lines()), CALL)[0]
    assert ((p.mean_intensity - p.background) < 3 * p.noise).all()     # old test: too dim
    assert not p.too_dim.any() and np.isfinite(p.score).all()
    assert ((p.ridge_intensity - p.background) >= 5 * p.noise).all()


def test_ridge_intensity_is_recorded_for_too_dim_patches_too():
    p = score_field(field_with(jm1=noise()), CALL)[0]
    assert p.too_dim.all() and np.isfinite(p.ridge_intensity).all()
    assert (p.ridge_intensity > p.mean_intensity).all()


def test_an_absolute_ridge_cut_replaces_the_noise_multiple():
    # sparse_lines: ridge pixels ~6.5-7 x noise (~13-15 counts) above the floor. A cut
    # above that calls every patch too dim; one below scores them, noise rule or not.
    cfg = MembraneScoreConfig
    base = score_field(field_with(jm1=sparse_lines()), cfg())[0]
    sig = base.ridge_intensity - base.background
    high = score_field(field_with(jm1=sparse_lines()), cfg(min_ridge_intensity=float(sig.max()) + 1))[0]
    low = score_field(field_with(jm1=sparse_lines()), cfg(min_ridge_intensity=float(sig.min()) - 1))[0]
    assert high.too_dim.all() and high.score.isna().all()
    assert not low.too_dim.any() and np.allclose(low.score, base.score)


def test_the_absolute_cut_overrides_the_noise_multiple_both_ways():
    # pure noise: ridge ~3.3 x noise, too dim under 5 x noise; a cut of 1 count scores it.
    p = score_field(field_with(jm1=noise()), MembraneScoreConfig(min_ridge_intensity=1.0))[0]
    assert not p.too_dim.any() and np.isfinite(p.score).all()


@pytest.mark.parametrize("fraction, value, floor", [(0.02, 50.0, 0.0), (0.02, 255.0, 3.0)])
def test_constant_specks_stay_too_dim_under_an_absolute_cut(fraction, value, floor):
    # Hot pixels lift the ridge mean far over any small cut, but there is no noise: no
    # signal above the floor, so the channel is still blank.
    cfg = MembraneScoreConfig(min_ridge_intensity=1.0)
    s = score_field(field_with(jm1=_specks(fraction, value, floor)), cfg)[0]
    assert s.too_dim.all() and s.score.isna().all()


def _specks(fraction, value, floor):
    """A constant ``floor`` with ``fraction`` of pixels, at random, at one constant ``value``."""
    img = np.full(N * N, floor, np.float32)
    img[np.random.default_rng(2).choice(N * N, int(N * N * fraction), replace=False)] = value
    return img.reshape(N, N)


@pytest.mark.parametrize("fraction, value, floor", [(0.02, 50.0, 0.0), (0.02, 255.0, 3.0),
                                                    (0.30, 50.0, 0.0)])
def test_constant_specks_on_a_constant_floor_are_too_dim(fraction, value, floor):
    # Isolated pixels of one value (hot or saturated specks) have no above-floor neighbour,
    # so they give no differences and no noise: the region-spread fallback (0 here)
    # decides and the channel reads as blank.
    s = score_field(field_with(jm1=_specks(fraction, value, floor)), CALL)[0]
    assert s.too_dim.all() and s.score.isna().all() and s.membrane_like.isna().all()


def test_score_is_invariant_to_intensity_scale():
    a = score_field(field_with(jm1=lines()), CALL)[0]
    b = score_field(field_with(jm1=lines() * 10), CALL)[0]
    assert np.allclose(a.score, b.score, rtol=1e-6) and (a.too_dim == b.too_dim).all()


@pytest.mark.parametrize("offset", [100.0, 1000.0])
@pytest.mark.parametrize("gain", [1.0, 5.0])
def test_score_is_invariant_to_an_additive_offset(offset, gain):
    # contrast subtracts the field floor from both means, so a camera offset or a uniform
    # background does not pull the score toward 1. Float32 input loses a little precision
    # under the offset: measured worst case 4e-7 relative, so 1e-5 leaves margin.
    a = score_field(field_with(jm1=lines() * gain), CALL)[0]
    b = score_field(field_with(jm1=lines() * gain + offset), CALL)[0]
    assert np.allclose(a.score, b.score, rtol=1e-5) and (a.too_dim == b.too_dim).all()


def test_same_input_gives_identical_scores():
    a, b = (score_field(field_with(jm1=lines()), CALL)[0] for _ in range(2))
    pd.testing.assert_frame_equal(a, b)


def test_nucleus_channel_is_never_scored_and_every_other_role_is():
    s = score_field(field_with(jm1=lines(), jm2=noise(), segment=lines()), CALL)[0]
    assert set(s.role) == {"JM1", "JM2", "segment"}


def test_a_blank_nucleus_plane_leaves_every_patch_unscored():
    p, f = score_field(field_with(jm1=lines(), nucleus=np.zeros((400, 400))), CALL)
    assert (p.region_fraction == 0).all() and p.score.isna().all() and (f.n_scored == 0).all()


def test_a_field_smaller_than_a_patch_does_not_crash():
    p, f = score_field(small_field(), CALL)
    assert len(f) >= 1 and (f.n_patches <= 1).all()


def test_coarse_pixels_clamp_the_filter_scale():
    # line_width_um / px / 2 = 0.15 to 0.3 px, clamped to 1 px. region_um 60 so the
    # radius-10 nuclei still grow to cover the field at 1 um per px.
    p, _ = score_field(field_with(jm1=lines(), px=1.0), MembraneScoreConfig(patch_um=200, region_um=60))
    assert (p.region_fraction >= 0.5).all() and not p.too_dim.any()
    assert np.isfinite(p.loc[~p.too_dim, "score"]).all()


@pytest.mark.parametrize("jm1, lo, hi", [(lines, 1.5, np.inf), (noise_on_floor, 0.95, 1.05)])
def test_the_downsampled_filter_path_scores_like_the_native_one(jm1, lo, hi):
    # At 0.075 um the smallest sigma is 2 px, so the filter runs on a 2x-downsampled plane.
    # The baseline must be shuffled before downsampling, or lone bright pixels at the
    # 1 px scale inflate it and thin lines fall below the cutoff. Patches are 267 px, so
    # only the top-left one is full; the 133 px edge patches are under half a side.
    p, _ = score_field(field_with(jm1=jm1(), px=0.075), CALL)
    full = p[(p.y0_px == 0) & (p.x0_px == 0)]
    assert not p.too_dim.any() and full.score.between(lo, hi).all()
    assert p.loc[(p.y0_px > 0) | (p.x0_px > 0), "score"].isna().all()


def test_an_edge_patch_narrower_than_a_filter_block_is_left_unscored():
    # 0.07 um: patches of 286 px, filter on 2x blocks. A 287 px wide field leaves a last
    # column of patches 1 px wide, which holds no whole block to filter.
    f = field_with(jm1=lines(), px=0.07)
    f.planes = {r: p[:, :287].copy() for r, p in f.planes.items()}
    p, _ = score_field(f, CALL)
    edge = p[p.x0_px == 286]
    assert len(edge) == 2 and edge.score.isna().all() and edge.membrane_like.isna().all()
    assert np.isfinite(p.loc[(p.x0_px == 0) & (p.y0_px == 0), "score"]).all()


def _widened(width):
    """The lines fixture with its columns repeated out to ``width`` px (same 50 px grid)."""
    f = field_with(jm1=lines())
    f.planes = {r: np.take(p, np.arange(width) % N, axis=1) for r, p in f.planes.items()}
    return f


def test_a_thin_edge_sliver_is_not_scored_and_does_not_dilute_the_fraction():
    # 404 px: the last patch column is 4 px wide, under half a 200 px patch side.
    p, f = score_field(_widened(404), CALL)
    sliver = p[p.x0_px == 400]
    assert len(sliver) == 2 and (sliver.region_fraction == 1).all()
    assert sliver.score.isna().all() and not sliver.too_dim.any() and sliver.membrane_like.isna().all()
    row = f.set_index("role").loc["JM1"]
    assert row.n_patches == 6 and row.n_scored == 4
    assert row.membrane_like_fraction == pytest.approx(1.0)


def test_a_clipped_patch_of_at_least_half_a_side_is_scored():
    # 500 px: the last patch column is 100 px wide, exactly half a side.
    p, _ = score_field(_widened(500), CALL)
    edge = p[p.x0_px == 400]
    assert len(edge) == 2 and np.isfinite(edge.score).all() and edge.membrane_like.all()


def test_the_fraction_is_weighted_by_region_pixels_not_region_fraction():
    # 500 px: two full columns (40000 region px per patch) and a half-width one (20000).
    # Make only the half-width column membrane-like: its share of the region is
    # 2 x 20000 / 200000 = 0.2, where counting patches by region_fraction would give 1/3.
    f = _widened(500)
    f.planes[Role.JM1][:, :400] = noise_on_floor()[:, :400]
    p, fields = score_field(f, CALL)
    assert p.loc[p.x0_px == 400, "membrane_like"].all() and not p.loc[p.x0_px < 400, "membrane_like"].any()
    assert fields.set_index("role").loc["JM1", "membrane_like_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize("value", [0.0, 255.0])
def test_a_constant_channel_is_too_dim_without_warnings(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, _ = score_field(field_with(jm1=np.full((400, 400), value)), CALL)
    assert p.too_dim.all() and p.membrane_like.isna().all()


def test_nan_pixels_are_ignored():
    img = lines(); img[::7, ::7] = np.nan
    p, _ = score_field(field_with(jm1=img), CALL)
    assert np.isfinite(p.score).all() and p.membrane_like.all()


# --- building blocks -----------------------------------------------------------------

def test_cell_region_grows_nuclei_by_region_um():
    nuc = np.zeros((100, 100), np.float32)
    nuc[45:55, 45:55] = 200.0
    region = cell_region(nuc, px_um=0.5, region_um=5.0)        # 10 px growth
    assert region[50, 50] and region[50, 64] and not region[50, 70] and not region[0, 0]


def test_cell_region_of_a_blank_nucleus_plane_is_empty():
    assert not cell_region(np.zeros((50, 50), np.float32), px_um=0.1, region_um=6.0).any()


def test_floor_and_noise_reads_percentile_and_robust_noise():
    background, noise_sd = floor_and_noise(noise())
    assert background == pytest.approx(50 - 2.33 * 2, abs=0.3)
    assert noise_sd == pytest.approx(2.0, rel=0.1)       # neighbour differences of white noise / sqrt(2)


@pytest.mark.parametrize("px", [0.0749, 0.075, 0.1, 0.2, 0.325, 1.0])
def test_filter_sigmas_stay_above_one_pixel(px):
    # skimage 0.26 hessian_matrix truncates its kernels at 8 sigma only when every sigma
    # is > 1, else at 100 sigma: about 10x slower. Exactly 1.0 must never reach the filter.
    factor, sigmas = _filter_geometry(px, MembraneScoreConfig().line_width_um)
    assert factor >= 1 and sigmas and min(sigmas) > 1.0


def test_floor_and_noise_of_a_constant_plane_is_zero_noise():
    assert floor_and_noise(np.full((50, 50), 7.0, np.float32)) == (7.0, 0.0)


def test_no_cutoff_makes_no_membrane_like_call():
    p, f = score_field(field_with(jm1=lines()), MembraneScoreConfig())
    assert np.isfinite(p.score).all() and p.membrane_like.isna().all()
    assert str(p.membrane_like.dtype) == "boolean" and f.membrane_like_fraction.isna().all()


def _dense_lines(height, pitch=9, width=2, seed=0):
    """Dense cells (pitch 9 px, about 3 um at 0.325 um/px): 2 px lines ``height`` on 10,
    blurred by a 0.6 px PSF, then camera noise sd 2 added after the blur, as in real data."""
    y, x = _yx()
    sig = ndi.gaussian_filter(np.where((y % pitch < width) | (x % pitch < width), height, 10.0), 0.6)
    return (sig + np.random.default_rng(seed).normal(0, 2, (N, N))).astype(np.float32)


@pytest.mark.parametrize("height", [30, 300, 3000])
def test_dense_line_structure_does_not_count_as_noise(height):
    # The old high-pass residual kept thin lines, so in dense fields its MAD measured the
    # lines (up to ~200x the true sd at height 3000). Noise must stay near the camera sd of
    # 2; at 9 px pitch line edges are a large share of pixel pairs, so the brightest grid
    # nudges it up (measured 2.8, 4.9, 9.4), still under 1% of the line height.
    assert floor_and_noise(_dense_lines(height))[1] < max(6.0, 0.01 * height)


def test_too_dim_does_not_change_when_dense_lines_get_brighter():
    cfg = MembraneScoreConfig(region_um=20)
    a, b = (score_field(field_with(jm1=_dense_lines(h), px=0.325), cfg)[0] for h in (30, 300))
    assert not a.too_dim.any() and not b.too_dim.any()


def test_dense_faint_lines_are_not_too_dim():
    cfg = MembraneScoreConfig(region_um=20)     # 50 px nuclei grown to cover the field
    p = score_field(field_with(jm1=_dense_lines(30), px=0.325), cfg)[0]
    assert not p.too_dim.any()


@pytest.mark.parametrize("sd", [0.3, 0.6])
def test_quantised_low_noise_lines_are_not_too_dim(sd):
    # 8-bit data with noise under one count: most neighbour differences are exactly 0, so
    # a plain MAD of them is 0 and the channel would read blank. Lines 25 counts up must score.
    y, x = _yx()
    sig = ndi.gaussian_filter(np.where(_borders(), 30.0, 5.0), 0.6)
    img = np.round(sig + np.random.default_rng(0).normal(0, sd, (N, N))).astype(np.uint8)
    p = score_field(field_with(jm1=img), CALL)[0]
    assert floor_and_noise(img)[1] > 0 and not p.too_dim.any()
