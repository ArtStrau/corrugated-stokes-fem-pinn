"""Apply native-library defaults before test modules import numerical code."""

import os

from scripts._native_environment import configure_native_library_environment


configure_native_library_environment(os.name, os.environ)
