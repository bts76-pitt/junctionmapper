"""Run configuration: which image channel fills each role, and which marker it carries.

Roles are slots in the analysis. The protein occupying a slot is configuration,
never hardcoded.
"""

from enum import StrEnum

import math

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from junctionmapper.errors import ConfigError

# Single definition of the geometry layer's reproducibility defaults. The geometry and QC
# modules read these rather than repeating the literals, so a change to a default cannot
# leave one call site behind.
DEFAULT_TCJ_MIN_RUN = 3
"""Shortest contact run the TCJ method keeps (``tcj_corners(min_run=...)``)."""

DEFAULT_NEIGHBOUR_RADIUS_PX = 2
"""Chebyshev radius of the window used to assign a boundary pixel its neighbour label."""


class Role(StrEnum):
    JM1 = "JM1"
    JM2 = "JM2"
    NUCLEUS = "nucleus"
    # The channel Cellpose draws cell outlines on. Never measured, and never the same
    # channel as JM1 or JM2: a boundary placed on a marker makes that marker's occupancy
    # ~100% by construction.
    SEGMENT = "segment"


class ChannelSpec(BaseModel):
    role: Role
    channel: str | int
    marker: str


class RunConfig(BaseModel):
    # A misspelled key is a mistake, not a comment: silently dropping it would mean running at
    # a default the operator believes they changed.
    model_config = ConfigDict(extra="forbid")

    channels: list[ChannelSpec]
    pixel_size_um: float | None
    dilation_px: int = 2
    dilation_um: float | None = None
    num_intra_cells: int = 5
    all_pairs: bool = False
    threshold: dict[Role, float] | None = None
    # Reproducibility parameters of the geometry layer; defaults are the values it uses.
    epsilon_px: float = 2.0                                 # Douglas-Peucker tolerance
    tcj_min_run: int = DEFAULT_TCJ_MIN_RUN                  # shortest TCJ contact run kept
    neighbour_radius_px: int = DEFAULT_NEIGHBOUR_RADIUS_PX  # neighbour-assignment window

    @field_validator("epsilon_px", mode="before")
    @classmethod
    def _check_epsilon(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ConfigError(f"epsilon_px must be a finite number > 0, got {v!r}")
        return v

    @field_validator("tcj_min_run", "neighbour_radius_px", mode="before")
    @classmethod
    def _check_positive_int(cls, v, info):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ConfigError(f"{info.field_name} must be an integer >= 1, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate(self) -> "RunConfig":
        roles = [c.role for c in self.channels]
        duplicates = sorted({r for r in roles if roles.count(r) > 1})
        if duplicates:
            raise ValueError(f"duplicate channel roles: {duplicates}")
        missing = [r for r in (Role.JM1, Role.NUCLEUS) if r not in roles]
        if missing:
            raise ValueError(f"required channel roles missing: {missing}")
        # A threshold for the nucleus role is never read (only JM1/JM2 are measured against
        # one), so accepting it silently would promise something that does not happen.
        for unmeasured in (Role.NUCLEUS, Role.SEGMENT):
            if self.threshold and unmeasured in self.threshold:
                raise ValueError("threshold is only used for the JM1 and JM2 roles; "
                                 f"remove the '{unmeasured.value}' entry")
        by_role = {c.role: c.channel for c in self.channels}
        if Role.SEGMENT in by_role:
            for measured in (Role.JM1, Role.JM2):
                if by_role.get(measured) == by_role[Role.SEGMENT]:
                    raise ValueError(
                        f"the segment channel {by_role[Role.SEGMENT]!r} is also the {measured.value} "
                        f"channel; segment on a marker that is not measured in the same run")
        return self

    def role_for(self, r: Role) -> ChannelSpec:
        for spec in self.channels:
            if spec.role == r:
                return spec
        raise KeyError(f"no channel configured for role {r.value}")

    def marker_for(self, r: Role) -> str:
        return self.role_for(r).marker

    def resolved_dilation_px(self, pixel_size_um: float | None = None) -> int:
        """Dilation band width in pixels.

        ``pixel_size_um`` (the field's actual pixel size) wins over the config
        fallback ``self.pixel_size_um`` when converting ``dilation_um``.
        """
        if self.dilation_um is None:
            return max(1, self.dilation_px)
        size = pixel_size_um if pixel_size_um is not None else self.pixel_size_um
        if not size or size <= 0:
            raise ConfigError("dilation_um requires a positive pixel size")
        return max(1, round(self.dilation_um / size))
