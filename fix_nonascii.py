"""
Replace non-ASCII special characters with ASCII equivalents in target Python source files.
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Target files (relative to script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_FILES = [
    'src/tasking_helper/utils/keplerian.py',
    'src/tasking_helper/utils/lwir.py',
    'src/tasking_helper/utils/moon_eci.py',
    'src/tasking_helper/utils/moon_jpl.py',
    'src/tasking_helper/utils/sun_eci.py',
    'src/tasking_helper/utils/sun_jpl.py',
    'src/tasking_helper/utils/__init__.py',
    'examples/ephemeris_to_tle.py',
    'examples/moon_ecr_states.py',
    'examples/sun_ecr_states.py',
    'examples/_test_keplerian.py',
    'examples/_test_lwir.py',
    'examples/_gen_test_ephem.py',
]

# Replacement mapping — ORDER MATTERS.
# 1. μm → um  (before μ → mu, so units stay correct)
# 2. Multi-char superscripts ⁻¹ ⁻² ⁻³ before ⁻ and individual digits
# 3. Everything else
REPLACEMENTS = [
    # --- μm / µm unit fix (must come before μ/µ → mu) ---
    # U+03BC (GREEK SMALL LETTER MU) + m
    ('μm', 'um'),
    # U+00B5 (MICRO SIGN) + m
    ('µm', 'um'),

    # --- Superscript compound forms (must come before ⁻ alone) ---
    ('⁻¹', '^-1'),
    ('⁻²', '^-2'),
    ('⁻³', '^-3'),

    # --- Greek letters ---
    ('λ', 'lam'),
    ('μ', 'mu'),       # U+03BC GREEK SMALL LETTER MU
    ('µ', 'mu'),       # U+00B5 MICRO SIGN
    ('Ω', 'Omega'),
    ('Σ', 'Sigma'),    # not in spec but present in files
    ('α', 'alpha'),
    ('β', 'beta'),
    ('ε', 'eps'),
    ('ζ', 'zeta'),     # not in spec but present in files
    ('π', 'pi'),
    ('σ', 'sigma'),
    ('θ', 'theta'),
    ('δ', 'delta'),
    ('Δ', 'Delta'),
    ('ν', 'nu'),
    ('ω', 'omega'),
    ('ψ', 'psi'),
    ('φ', 'phi'),
    ('η', 'eta'),

    # --- Superscripts / subscripts (remaining, after compound forms handled above) ---
    ('²', '^2'),
    ('³', '^3'),
    ('⁴', '^4'),
    ('⁵', '^5'),
    ('⁶', '^6'),
    ('⁷', '^7'),
    ('⁸', '^8'),
    ('⁹', '^9'),
    ('¹', '^1'),       # U+00B9 superscript one (found in keplerian.py)
    ('⁻', '^-'),       # catch remaining superscript minus
    ('₀', '_0'),
    ('₁', '_1'),
    ('₂', '_2'),
    ('₃', '_3'),

    # --- Math / relation symbols ---
    ('≈', '~='),
    ('≲', '<~'),
    ('≡', '=='),
    ('≠', '!='),
    ('≤', '<='),
    ('≥', '>='),
    ('→', '->'),
    ('←', '<-'),
    ('↔', '<->'),
    ('×', 'x'),
    ('÷', '/'),
    ('·', '*'),
    ('±', '+/-'),
    ('∞', 'inf'),
    ('∫', 'integral'),
    ('√', 'sqrt'),
    ('∝', 'proportional_to'),  # not in spec but present in lwir.py
    ('∈', 'in'),               # not in spec but present in files
    ('≪', '<<'),               # not in spec but present in lwir.py
    ('−', '-'),                # U+2212 minus sign

    # --- Dashes ---
    ('–', '-'),                # U+2013 en dash
    ('—', '--'),               # U+2014 em dash

    # --- Box-drawing characters ---
    ('─', '-'),                # U+2500
    ('│', '|'),                # U+2502
    ('└', '+'),
    ('┌', '+'),

    # --- Misc ---
    ('°', 'deg'),
    ('…', '...'),
    ('′', "'"),
    ('″', '"'),

    # --- Other found characters not in original spec ---
    ('§', 'sec'),              # section sign (found in keplerian.py, likely "section")
    ('̂', ''),                  # U+0302 combining circumflex accent — drop it (it's a diacritic on prev char)
]


def process_file(rel_path):
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        print(f"  SKIP (not found): {rel_path}")
        return

    with open(abs_path, 'r', encoding='utf-8') as fh:
        original = fh.read()

    content = original
    replacements_made = {}  # char → count

    for old, new in REPLACEMENTS:
        if old in content:
            count = content.count(old)
            content = content.replace(old, new)
            replacements_made[old] = replacements_made.get(old, 0) + count

    if replacements_made:
        with open(abs_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        total = sum(replacements_made.values())
        print(f"\n  {rel_path}  ({total} replacements)")
        for char, cnt in replacements_made.items():
            desc = repr(char)
            replacement = dict(REPLACEMENTS)[char]
            print(f"    {desc} (U+{ord(char[0]):04X})  x{cnt}  ->  {repr(replacement)}")
    else:
        print(f"\n  {rel_path}  (no non-ASCII characters found)")


def verify_file(rel_path):
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        return None
    with open(abs_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    non_ascii = [(hex(ord(c)), c) for c in set(content) if ord(c) > 127]
    return non_ascii


print("=" * 70)
print("PASS 1: Replacing non-ASCII characters")
print("=" * 70)

for f in TARGET_FILES:
    process_file(f)

print("\n")
print("=" * 70)
print("PASS 2: Verification — checking for remaining non-ASCII characters")
print("=" * 70)

all_clean = True
for f in TARGET_FILES:
    remaining = verify_file(f)
    if remaining is None:
        print(f"  {f}: MISSING")
    elif remaining:
        all_clean = False
        print(f"  {f}: STILL HAS NON-ASCII: {remaining}")
    else:
        print(f"  {f}: CLEAN")

print()
if all_clean:
    print("All target files are now ASCII-clean.")
else:
    print("WARNING: Some files still contain non-ASCII characters (see above).")
