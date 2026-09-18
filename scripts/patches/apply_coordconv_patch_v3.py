"""
apply_coordconv_patch_v3.py

Fixes v2's bug: the conditional stem_conv block must be its own statement
BEFORE `self.stem = nn.Sequential(`, not inside that call's argument list.

Run from the repo root:
    python apply_coordconv_patch_v3.py
"""
import re

with open('src/model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

class_idx = None
for i, l in enumerate(lines):
    if l.startswith('class AnnulusPlaneNet'):
        class_idx = i
        break
assert class_idx is not None, "Could not find 'class AnnulusPlaneNet' — aborting, no changes made"

init_idx = None
for i in range(class_idx, min(class_idx + 30, len(lines))):
    if 'def __init__' in lines[i] and 'use_cbam' in lines[i]:
        init_idx = i
        break
assert init_idx is not None, "Could not find AnnulusPlaneNet.__init__ signature — aborting, no changes made"

old_init_line = lines[init_idx]
assert 'use_coordconv' not in old_init_line, "Patch already applied — no changes made."

usecbam_idx = None
for i in range(init_idx, min(init_idx + 10, len(lines))):
    if re.search(r'self\.use_cbam\s*=\s*use_cbam', lines[i]):
        usecbam_idx = i
        break
assert usecbam_idx is not None, "Could not find 'self.use_cbam = use_cbam' — aborting, no changes made"

# Find "self.stem = nn.Sequential(" line
stem_idx = None
for i in range(usecbam_idx, min(usecbam_idx + 15, len(lines))):
    if re.search(r'self\.stem\s*=\s*nn\.Sequential\(', lines[i]):
        stem_idx = i
        break
assert stem_idx is not None, "Could not find 'self.stem = nn.Sequential(' — aborting, no changes made"

# Find the nn.Conv3d(1, 32, kernel_size=7...) line AFTER stem_idx (inside the Sequential call)
conv_idx = None
for i in range(stem_idx, min(stem_idx + 10, len(lines))):
    if re.search(r'nn\.Conv3d\(1,\s*32,\s*kernel_size=7', lines[i]):
        conv_idx = i
        break
assert conv_idx is not None, "Could not find stem's nn.Conv3d(1, 32, kernel_size=7...) line — aborting, no changes made"

stem_line_indent = lines[stem_idx][:len(lines[stem_idx]) - len(lines[stem_idx].lstrip())]
inner_indent = lines[conv_idx][:len(lines[conv_idx]) - len(lines[conv_idx].lstrip())]

# --- Apply edits bottom-to-top so earlier indices stay valid ---

# (d) replace the nn.Conv3d(1,32,...) line (inside Sequential) with just "stem_conv,"
lines[conv_idx:conv_idx+1] = [f"{inner_indent}stem_conv,\n"]

# (c) insert the conditional block BEFORE self.stem = nn.Sequential(
conditional_block = [
    f"{stem_line_indent}if use_coordconv:\n",
    f"{stem_line_indent}    from src.coordconv3d import CoordConv3D\n",
    f"{stem_line_indent}    stem_conv = CoordConv3D(1, 32, kernel_size=7, stride=2, padding=3,\n",
    f"{stem_line_indent}                            bias=False, with_r=coordconv_r)\n",
    f"{stem_line_indent}else:\n",
    f"{stem_line_indent}    stem_conv = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)\n",
]
lines[stem_idx:stem_idx] = conditional_block

# (b) insert extra assignment lines after usecbam_idx
extra_assign_lines = [
    "        self.use_coordconv = use_coordconv\n",
    "        self.coordconv_r   = coordconv_r\n",
]
lines[usecbam_idx+1:usecbam_idx+1] = extra_assign_lines

# (a) replace the init signature line with the two-line version
new_lines_block = [
    "    def __init__(self, dropout_p: float = 0.3, use_se: bool = True, use_cbam: bool = False,\n",
    "                 use_coordconv: bool = False, coordconv_r: bool = False):\n",
]
lines[init_idx:init_idx+1] = new_lines_block

with open('src/model.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Patch v3 applied successfully.")
print("Verify with the model-loading check before relying on this.")