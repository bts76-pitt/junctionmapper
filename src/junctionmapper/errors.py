"""Exception hierarchy for junctionmapper."""


class JunctionMapperError(Exception):
    """Base class for all errors raised by junctionmapper."""


class ConfigError(JunctionMapperError, ValueError):
    """Invalid or insufficient run configuration."""


class CalibrationError(JunctionMapperError):
    """No physical pixel size is available from metadata or configuration."""


class ChannelError(JunctionMapperError):
    """A configured channel cannot be resolved against the image."""
