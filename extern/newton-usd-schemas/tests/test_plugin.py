# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import pathlib
import unittest

from pxr import Plug

import newton_usd_schemas


class TestNewtonPlugin(unittest.TestCase):
    def test_newton_plugin_registered(self):
        plugin = Plug.Registry().GetPluginWithName("newton")
        self.assertIsInstance(plugin, Plug.Plugin)
        self.assertEqual(plugin.resourcePath, pathlib.Path(newton_usd_schemas.__file__).parent.as_posix())


if __name__ == "__main__":
    unittest.main()
