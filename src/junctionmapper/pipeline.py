"""Run the measurement over one field and emit tidy tables.

Core library functions return pixels; this module is where scaling to micrometres happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from junctionmapper.config import Role, RunConfig
from junctionmapper.errors import ConfigError
from junctionmapper.geometry.interfaces import LOW_CONTACT_FRACTION, extract_interfaces
from junctionmapper.io.reader import Field
from junctionmapper.measure import interface as mi
from junctionmapper.measure.nuclei import associate, intra_distances, nucleus_centroids
from junctionmapper.qc import assess_cell
from junctionmapper.segment.cellpose import Segmentation
from junctionmapper.tables import make_frame

METHODS = ("dp", "tcj")

# ---- thresholds ---------------------------------------------------------------------
# The design (spec, "Intensity thresholds") asks for "a single value derived from a
# calibration subset [...] carried across the run, so the common path is 'accept', with
# per-field override where review shows it is needed". Per-field Otsu is not that: Otsu on
# two near-identical fields returns 40.234375 and 40.341797, which made the report's
# "threshold varied, not comparable" banner fire on every multi-field run and so taught the
# operator to ignore the one warning that matters.

CALIBRATION_FIELDS = 5
"""How many fields the run threshold is proposed from (see ``calibration_subset``)."""

THRESHOLD_SAME_TOL = 1.0
"""Grey levels within which two APPLIED thresholds count as the same value.

The Java's threshold control is an integer slider on a 0-255 scale, so one grey level is the
smallest difference an operator could have asked for; anything smaller is float noise, not an
override. Used by the report to decide whether a threshold genuinely varied across fields.
"""

THRESHOLD_DEVIATION_FRACTION = 0.10
THRESHOLD_DEVIATION_FLOOR = 1.0
"""Tolerance for ``threshold_deviates_from_run``: 10% of the applied value, never less than
one grey level.

Relative rather than a flat grey-level bound because the planes are not necessarily 8-bit: a
16-bit marker plane has Otsu values in the thousands, where any absolute bound chosen for
8-bit data would flag every field or none. The floor keeps it sane as T approaches 0, and is
one grey level for the reason given above (it also covers skimage's 256-bin quantisation of
Otsu, which is one bin = range/256). 10% is the scale at which the measured marker area
changes materially, T sitting in the shoulder of the intensity distribution.

On a LOW-VALUED plane the floor dominates and the flag fires on an ordinary DEFAULT run: with
a run value of 2.28, 10% is 0.23 and the floor of 1.0 is what applies, so a field whose own
Otsu is 1.1 grey levels away is flagged although nobody overrode anything. That is the
intended reading -- against a threshold of 2.28 one grey level really is a large relative
disagreement -- and it is why the flag is informational and never an error. Do not retune the
floor to hide it: the floor is what stops a 16-bit plane tripping on noise.
"""


def _pair_id(field_id: str, method: str, cell_id: int, neighbour_id: int) -> str:
    """Stable, order-independent key for the CELL PAIR an interface belongs to.

    Format: ``field|method|lo-hi``, with the two cell ids sorted.

    The row measured from A against B and the row measured from B against A
    therefore carry the same ``pair_id``, and can be grouped or deduplicated
    without guessing.

    The method is part of the key because dp and tcj partition the boundary
    differently, so their interfaces are not the same objects.

    Several contact runs between one pair share the pair id, so deduplicating by
    it counts the junction once whatever the run count. ``interface_id`` is the
    unique key of one row's interface.
    """
    lo, hi = sorted((int(cell_id), int(neighbour_id)))
    return f"{field_id}|{method}|{lo}-{hi}"


def _interface_id(field_id: str, method: str, cell_id: int, start_idx: int) -> str:
    """Unique key of one measured cell-side interface (both roles share it)."""
    return f"{field_id}|{method}|{int(cell_id)}@{int(start_idx)}"


@dataclass
class FieldResult:
    interfaces: pd.DataFrame
    cells: pd.DataFrame
    nuclei: pd.DataFrame
    flags: pd.DataFrame
    nuclei_distances: pd.DataFrame


@dataclass(frozen=True)
class RunThresholds:
    """The threshold applied to every field of one run, per marker role.

        applied              the value used
        source[role]         where it came from (below)
        calibration_fields   the subset that was used
        calibration_values   the Otsu values that subset contributed, per role

    ``source[role]`` is one of:

        "config"        from ``cfg.threshold``
        "calibration"   the median over the calibration subset
        "flat"          every calibration field was flat or non-finite, so 0.0
    """
    applied: dict[Role, float]
    source: dict[Role, str]
    calibration_fields: tuple[str, ...]
    calibration_values: dict[Role, tuple[float, ...]]

    def describe(self) -> str:
        """One human line per role, for provenance and the report."""
        parts = []
        for role in (Role.JM1, Role.JM2):
            if role not in self.applied:
                continue
            src, vals = self.source[role], self.calibration_values.get(role, ())
            if src == "config":
                why = "from the run configuration"
            elif src == "flat":
                why = (f"every calibration field ({', '.join(self.calibration_fields)}) had a "
                       f"flat or non-finite plane, so Otsu could not run and T fell back to 0.0")
            else:
                why = (f"median of Otsu on {len(vals)} calibration field(s) "
                       f"({', '.join(self.calibration_fields)}): "
                       f"{', '.join(format(v, '.6g') for v in vals)}")
            parts.append(f"{role.value} = {self.applied[role]:.6g} ({why})")
        return "; ".join(parts) or "no marker role resolved"


def otsu_threshold(plane) -> float | None:
    """Otsu's threshold of one marker plane, or None when Otsu cannot run.

    None means no threshold separates anything: the plane is empty, all
    non-finite, or flat (a single finite value).

    Callers decide what to do with that. The per-field fallback is 0.0, and the
    calibration subset excludes such planes.
    """
    vals = np.asarray(plane, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or vals.min() == vals.max():
        return None
    try:
        return float(threshold_otsu(vals))
    except ValueError:
        return None


def field_otsu(field: Field) -> dict[Role, float | None]:
    """Otsu per marker role present in ``field`` (None where Otsu cannot run)."""
    return {role: otsu_threshold(field.planes[role])
            for role in (Role.JM1, Role.JM2) if role in field.planes}


def calibration_subset(field_ids, n: int = CALIBRATION_FIELDS) -> list[str]:
    """Up to ``n`` field ids spread evenly over ``sorted(field_ids)``.

    The ids are sorted, then ``n`` evenly spaced positions are taken from the
    first to the last. With ``n`` or fewer ids, all of them come back.

    Deterministic, and independent of the order the fields were read.

    (Why spread rather than take the first ``n``: a plate acquired in order is
    then sampled from its start, middle and end.)
    """
    ids = sorted(set(map(str, field_ids)))
    if len(ids) <= n:
        return ids
    return [ids[int(round(i))] for i in np.linspace(0, len(ids) - 1, n)]


def propose_run_thresholds(fields, cfg: RunConfig) -> RunThresholds:
    """One threshold per marker role for the WHOLE run, from a calibration subset.

    Precedence, per role:

        1. ``cfg.threshold[role]`` wins outright.
        2. Otherwise, the MEDIAN of Otsu over the calibration subset.

    The subset is up to ``CALIBRATION_FIELDS`` (5) fields chosen by
    ``calibration_subset``. Median rather than mean, so one saturated or empty
    field cannot drag the run value.

    A field whose plane is flat or non-finite contributes nothing, since Otsu
    cannot run on it. If every calibration field is like that, the run value
    falls back to 0.0 and ``source[role]`` records ``"flat"``.

    ``fields`` is an iterable of ``Field``, consumed once.

    Cost
        The subset is chosen FIRST, and Otsu runs only on the fields in it. Cost
        is bounded by ``CALIBRATION_FIELDS``, not by how many fields the caller
        hands over -- one multi-scene or multi-Z file can hold a dozen.

        The price is that the fields handed over are held until the subset is
        known, because the subset is defined over the full sorted id set and so
        cannot be decided while streaming. Pass a calibration pass's handful of
        files, as the CLI does, not every field of a run.
    """
    held = {str(f.field_id): f for f in fields}
    chosen = calibration_subset(held)
    by_field = {fid: field_otsu(held[fid]) for fid in chosen}
    applied: dict[Role, float] = {}
    source: dict[Role, str] = {}
    values: dict[Role, tuple[float, ...]] = {}
    for role in (Role.JM1, Role.JM2):
        if cfg.threshold and role in cfg.threshold:
            applied[role], source[role], values[role] = float(cfg.threshold[role]), "config", ()
            continue
        if not any(role in by_field[f] for f in chosen):
            continue                            # the role has no plane in this run
        vs = tuple(by_field[f][role] for f in chosen
                   if by_field[f].get(role) is not None)
        values[role] = vs
        applied[role] = float(np.median(vs)) if vs else 0.0
        source[role] = "calibration" if vs else "flat"
    return RunThresholds(applied, source, tuple(chosen), values)


def threshold_deviates(own: float | None, applied: float) -> bool:
    """True when a field's own Otsu is far enough from the applied value to be worth a look.

    Tolerance:

        max(THRESHOLD_DEVIATION_FLOOR,
            THRESHOLD_DEVIATION_FRACTION * |applied|)

    A field whose own plane cannot be thresholded (``own is None``) never
    deviates, because there is nothing to compare.

    This is NOT a flag that only a configured or overridden threshold can trip.
    Where a plane's values are small the absolute floor dominates, so the outer
    fields of an ordinary default run are flagged too -- one value proposed and
    carried, nothing overridden.

    Informational by design. It says this field's own plane would have been
    thresholded elsewhere, and nothing more.
    """
    if own is None:
        return False
    tol = max(THRESHOLD_DEVIATION_FLOOR, THRESHOLD_DEVIATION_FRACTION * abs(float(applied)))
    return bool(abs(float(own) - float(applied)) > tol)


def resolve_thresholds(field: Field, cfg: RunConfig,
                       run: RunThresholds | None = None) -> dict[Role, float]:
    """Threshold T per marker role present in ``field``, in precedence order.

        1. ``cfg.threshold[role]``   an explicit operator override, per role
        2. ``run.applied[role]``     the run value from the calibration subset;
                                     the default path for a multi-field run
                                     (see ``propose_run_thresholds``)
        3. Otsu on this field's own plane, or 0.0 when Otsu cannot run

    Step 3 is the single-field fallback. With one field the calibration subset
    IS that field, so steps 2 and 3 agree.
    """
    out: dict[Role, float] = {}
    for role in (Role.JM1, Role.JM2):
        plane = field.planes.get(role)
        if plane is None:
            continue
        if cfg.threshold and role in cfg.threshold:
            out[role] = float(cfg.threshold[role])
            continue
        if run is not None and role in run.applied:
            out[role] = float(run.applied[role])
            continue
        t = otsu_threshold(plane)
        out[role] = 0.0 if t is None else t
    return out


def measure_field(field: Field, seg: Segmentation, cfg: RunConfig,
                  epsilon: float | None = None,
                  run_thresholds: RunThresholds | None = None) -> FieldResult:
    """Measure every non-border, non-degenerate cell of ``field``.

    Both corner methods are run. JM1 and, if present, JM2 are measured
    identically.

        epsilon                   ``cfg.epsilon_px`` unless ``epsilon`` is passed;
                                  the value used is recorded in ``cells.epsilon``
        cfg.tcj_min_run           passed to the geometry layer, recorded in ``cells``
        cfg.neighbour_radius_px   same

    ``run_thresholds`` is the run-level proposal from ``propose_run_thresholds``.
    Passing it applies one threshold per role to every field of the run. Without
    it, the field falls back to Otsu on its own plane.

    Either way, every interface row records all three of:

        threshold                     the value actually applied
        threshold_field_otsu          what this field's own plane would have given
        threshold_deviates_from_run   set when those two are far apart

    (Why all three: that is what makes a bad run diagnosable after the fact.)
    """
    if epsilon is None:
        epsilon = cfg.epsilon_px
    labels = np.asarray(seg.cells)
    nuc_labels = np.asarray(seg.nuclei)
    if labels.ndim != 2:
        raise ConfigError(f"cell labels must be 2-D, got shape {labels.shape}")
    for role, plane in field.planes.items():
        if np.shape(plane) != labels.shape:
            raise ConfigError(f"{role.value} plane shape {np.shape(plane)} != label shape {labels.shape}")
    px = float(field.pixel_size_um)
    fid = field.field_id
    cycles = cfg.resolved_dilation_px(px)
    thresholds = resolve_thresholds(field, cfg, run_thresholds)
    own_otsu = field_otsu(field)
    roles = [r for r in (Role.JM1, Role.JM2) if r in field.planes and r in thresholds
             and any(c.role == r for c in cfg.channels)]

    if nuc_labels.shape != labels.shape:
        raise ConfigError(f"nucleus labels shape {nuc_labels.shape} != cell labels {labels.shape}")
    mapping, nflags = associate(labels, nuc_labels)
    nuc_of = {n: c for c, ns in mapping.items() for n in ns}

    iface_rows, cell_rows, flag_rows = [], [], []
    for cid in (int(c) for c in np.unique(labels) if c != 0):
        fl = assess_cell(labels, cid, epsilon, cfg.tcj_min_run, cfg.neighbour_radius_px)
        area = int((labels == cid).sum())
        counts = {m: 0 for m in METHODS}
        nb_by_method: dict[str, set[int]] = {m: set() for m in METHODS}
        low_contact = {m: False for m in METHODS}
        closed_loop_cell = False
        measured = not (fl.border_touching or fl.degenerate)
        if measured:
            for method in METHODS:
                for itf in extract_interfaces(labels, cid, method, epsilon,
                                           cfg.tcj_min_run, cfg.neighbour_radius_px):
                    counts[method] += 1
                    nb_by_method[method].add(int(itf.neighbour_id))
                    low = itf.contact_fraction < LOW_CONTACT_FRACTION
                    low_contact[method] = low_contact[method] or low
                    closed = itf.start_idx == itf.end_idx
                    closed_loop_cell = closed_loop_cell or closed
                    pix = itf.pixels
                    contour = mi.interface_contour(pix)
                    straight = mi.straight_line_length(pix)
                    lin = mi.linearity_index(pix)
                    band = mi.dilated_band(pix, labels.shape, cycles)
                    for role in roles:
                        img, t = field.planes[role], thresholds[role]
                        bm = mi.band_metrics(band, img, t)
                        frag = mi.fragmented_contour(pix, img, t)
                        jc = mi.junction_contour(pix, img, t)
                        jsl = mi.junction_straight_line(pix, img, t)
                        iface_rows.append({
                            "field_id": fid, "cell_id": cid, "neighbour_id": itf.neighbour_id,
                            "corner_method": itf.method, "role": role.value,
                            "marker": cfg.marker_for(role),
                            "start_idx": itf.start_idx, "end_idx": itf.end_idx,
                            "pair_id": _pair_id(fid, itf.method, cid, itf.neighbour_id),
                            "interface_id": _interface_id(fid, itf.method, cid, itf.start_idx),
                            "contact_fraction": itf.contact_fraction,
                            "low_contact_fraction": low, "closed_loop": closed,
                            "threshold": t,
                            "threshold_field_otsu": (float("nan") if own_otsu.get(role) is None
                                                     else own_otsu[role]),
                            "threshold_deviates_from_run": threshold_deviates(own_otsu.get(role), t),
                            "dilation_cycles": cycles,
                            "interface_contour_px": contour, "interface_contour_um": contour * px,
                            "straight_line_length_px": straight,
                            "straight_line_length_um": straight * px,
                            "linearity_index": lin,
                            "fragmented_contour_px": frag, "fragmented_contour_um": frag * px,
                            "junction_contour_px": jc, "junction_contour_um": jc * px,
                            "junction_straight_line_px": jsl,
                            "junction_straight_line_um": jsl * px,
                            "coverage_index": mi.coverage_index(pix, img, t),
                            "interface_area_px": bm.interface_area_px,
                            "interface_area_um2": bm.interface_area_px * px * px,
                            "marker_area_px": bm.marker_area_px,
                            "marker_area_um2": bm.marker_area_px * px * px,
                            "marker_intensity": bm.marker_intensity,
                            "interface_occupancy_pct": bm.interface_occupancy_pct,
                            "intensity_per_px2": bm.intensity_per_area,
                            "intensity_per_um2": bm.intensity_per_area / (px * px),
                            "cluster_density_per_px2": bm.cluster_density,
                            "cluster_density_per_um2": bm.cluster_density / (px * px),
                        })
        neighbours = nb_by_method["dp"] | nb_by_method["tcj"]
        cell_rows.append({
            "field_id": fid, "cell_id": cid, "area_px": area, "area_um2": area * px * px,
            "measured": measured, "n_neighbours": len(neighbours),
            "n_interfaces_dp": counts["dp"], "n_interfaces_tcj": counts["tcj"],
            "n_nuclei": len(mapping.get(cid, [])), "epsilon": float(epsilon),
            "tcj_min_run": cfg.tcj_min_run, "neighbour_radius_px": cfg.neighbour_radius_px,
            "dilation_cycles": cycles, "pixel_size_um": px,
        })
        flag_rows.append({
            "field_id": fid, "cell_id": cid, "border_touching": fl.border_touching,
            "isolated": fl.isolated, "degenerate": fl.degenerate,
            "corner_methods_disagree": fl.corner_methods_disagree,
            "n_corners_dp": fl.n_corners_dp, "n_corners_tcj": fl.n_corners_tcj,
            "all_contacts_filtered": fl.all_contacts_filtered,
            "dp_low_contact": low_contact["dp"],
            "dp_missed_tcj_contact": bool(nb_by_method["tcj"] - nb_by_method["dp"]),
            "closed_loop_interface": closed_loop_cell,
            "multinucleate": cid in nflags.multinucleate,
            "anucleate": cid in nflags.anucleate_cells,
        })

    # Nuclei: ``nuclei_distances`` is LONG - one row per (nucleus, other nucleus) ORDERED pair.
    # Every nucleus owns its rows, so ``rank_from_nucleus`` is that nucleus's own ordering over
    # every distance involving it, in either direction, and ``rank_from_nucleus == 1`` is its
    # nearest neighbour for every nucleus, including the highest id. With ``all_pairs`` this
    # means each unordered pair appears twice, once from each end, each with its own rank; that
    # is the price of making the obvious selection (``rank_from_nucleus == 1``) correct, and it
    # is the shape the k-nearest mode already has (there a mutual nearest pair appears twice
    # too). A consumer wanting each distance once keeps the rows with
    # ``nucleus_id < neighbour_nucleus_id``. The summary columns in ``nuclei`` are the count and
    # mean of that nucleus's own rows, so they are a plain groupby of this table.
    cents = nucleus_centroids(nuc_labels)
    pairs: dict[int, list[tuple[float, int]]] = {}
    for i, j, d in intra_distances(cents, cfg.num_intra_cells, cfg.all_pairs):
        pairs.setdefault(i, []).append((d, j))
        if cfg.all_pairs:     # stored once there, but it is a distance of BOTH nuclei
            pairs.setdefault(j, []).append((d, i))
    dists: dict[int, list[float]] = {}
    dist_rows = []
    for i in sorted(pairs):
        for rank, (d, j) in enumerate(sorted(pairs[i]), start=1):     # nearest first
            dists.setdefault(i, []).append(d)
            dist_rows.append({
                "field_id": fid, "nucleus_id": i, "neighbour_nucleus_id": j,
                "rank_from_nucleus": rank, "distance_px": d, "distance_um": d * px,
            })
    nuc_rows = []
    for nid, (r, c) in cents.items():
        ds = dists.get(nid, [])
        mean = float(np.mean(ds)) if ds else float("nan")
        area = int((nuc_labels == nid).sum())
        nuc_rows.append({
            "field_id": fid, "nucleus_id": nid, "cell_id": nuc_of.get(nid),
            "orphan": nid not in nuc_of, "area_px": area, "area_um2": area * px * px,
            "centroid_row": r, "centroid_col": c, "n_intra_distances": len(ds),
            "mean_intra_distance_px": mean, "mean_intra_distance_um": mean * px,
        })

    return FieldResult(
        interfaces=make_frame("interfaces", iface_rows), cells=make_frame("cells", cell_rows),
        nuclei=make_frame("nuclei", nuc_rows), flags=make_frame("flags", flag_rows),
        nuclei_distances=make_frame("nuclei_distances", dist_rows),
    )
