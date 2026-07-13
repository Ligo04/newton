# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import pathlib

from ._version import __version__

__all__ = ["__version__"]

try:
    from pxr import Plug
except ImportError:  # pragma: no cover
    raise ImportError("OpenUSD python modules must be installed to use newton_usd_schemas")  # pragma: no cover

# register the newton schema plugin
Plug.Registry().RegisterPlugins([(pathlib.Path(__file__).parent).absolute().as_posix()])
