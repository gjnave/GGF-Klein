"""Mechanical extraction of city96 inference helpers; excludes its UI nodes."""
import ast
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
target = Path(__file__).resolve().parents[1] / "vendor" / "gguf_core"
target.mkdir(parents=True, exist_ok=True)
for name in ("ops.py", "loader.py", "dequant.py", "LICENSE"):
    shutil.copy2(source / name, target / name)
text = (source / "nodes.py").read_text(encoding="utf-8")
tree = ast.parse(text)
patcher = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GGUFModelPatcher")
header = '''# Extracted without changes from city96/ComfyUI-GGUF nodes.py (Apache-2.0).
import collections
import torch
import logging
import comfy.model_patcher
import comfy.model_management
import comfy.lora
import comfy.float
import comfy.utils
from .ops import move_patch_to_device
from .dequant import is_quantized, is_torch_compatible
'''
(target / "patcher.py").write_text(header + "\n" + ast.get_source_segment(text, patcher) + "\n", encoding="utf-8")
