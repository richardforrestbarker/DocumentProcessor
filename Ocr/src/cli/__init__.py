"""
CLI module for Receipt OCR Service.

Provides command-line interface for processing receipt images.
"""

from .args import create_argument_parser, parse_args
from .commands import process_command, version_command
from .utils import check_dependencies, get_device

__all__ = [
    'create_argument_parser',
    'parse_args',
    'process_command',
    'version_command',
    'check_dependencies',
    'get_device',
]
