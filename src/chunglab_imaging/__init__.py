"""Imaging tools for the Chung Lab at Harvard"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chunglab-imaging")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@example.com"
