"""Run configuration: which image channel fills each role, and which marker it carries.

Roles are slots in the analysis. The protein occupying a slot is configuration,
never hardcoded.
"""

from enum import StrEnum

from pydantic import BaseModel, model_validator

from junctionmapper.errors import ConfigError


class Role(StrEnum):
    JM1 = "JM1"
    JM2 = "JM2"
    NUCLEUS = "nucleus"


class ChannelSpec(BaseModel):
    role: Role
    channel: str | int
    marker: str


class RunConfig(BaseModel):
    channels: list[ChannelSpec]
    pixel_size_um: float | None
    dilation_px: int = 2
    dilation_um: float | None = None
    num_intra_cells: int = 5
    all_pairs: bool = False
    threshold: dict[Role, float] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RunConfig":
        roles = [c.role for c in self.channels]
        duplicates = sorted({r for r in roles if roles.count(r) > 1})
        if duplicates:
            raise ValueError(f"duplicate channel roles: {duplicates}")
        missing = [r for r in (Role.JM1, Role.NUCLEUS) if r not in roles]
        if missing:
            raise ValueError(f"required channel roles missing: {missing}")
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
