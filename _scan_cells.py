"""
_scan_cells.py
Scans all code cells in the notebook for literal newlines inside string literals
(the pattern that causes SyntaxError: unterminated string literal).
Prints cell index and the offending lines.
"""
import json, re

NB_PATH = "ML_Ex01_EDA.ipynb"
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

issues = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    # Find lines where a quoted string appears to be opened but not closed
    # (crude heuristic: look for ' + " or " + " that spans a newline in source list)
    lines = cell["source"]
    for j, line in enumerate(lines):
        # A line ending with a string that doesn't close is usually missing a quote
        stripped = line.rstrip("\n")
        # Count unescaped quotes
        single_open = stripped.count("'") % 2 != 0
        double_open = stripped.count('"') % 2 != 0
        # Only flag if there's an actual open paren on this line (inside a call)
        if (single_open or double_open) and ("set_title" in stripped or "set_xlabel" in stripped or "set_ylabel" in stripped or "suptitle" in stripped):
            issues.append((i, j, stripped))

if issues:
    print("POTENTIAL STRING LITERAL ISSUES:")
    for cell_idx, line_idx, content in issues:
        print("  Cell {:02d}, line {:02d}: {}".format(cell_idx, line_idx, content[:100]))
else:
    print("No obvious string literal issues found in plot calls.")

print("Total cells scanned:", len(nb["cells"]))
