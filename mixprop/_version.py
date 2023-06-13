__all__ = ['__version__']

# major, minor
version_info = 1, 0

# suffix
suffix = None

# version string
__version__ = '.'.join(map(str, version_info)) + (f'.{suffix}' if suffix else '')
