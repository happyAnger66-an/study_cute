# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Backward-compatible shim — implementation lives in fmha_d256/{host,device}/.

See fmha_d256/README.md for the package layout.
"""

from fmha_d256 import MixedInputFusedMultiHeadAttentionPrefillD256, run

__all__ = ["MixedInputFusedMultiHeadAttentionPrefillD256", "run"]
