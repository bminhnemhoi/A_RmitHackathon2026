"""Convert a '# %%' percent-format .py script into a Jupyter notebook (for Kaggle upload).

Usage: python scripts/py_to_ipynb.py notebooks/train_kaggle.py notebooks/RMIT_SGS_Fprinciples_C1.ipynb
"""
import json
import re
import sys


def convert(src_path: str, dst_path: str) -> int:
    src = open(src_path, encoding="utf-8").read()
    parts = re.split(r"^# %%(.*)$", src, flags=re.M)
    cells = []
    for tag, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if "[markdown]" in tag:
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        else:
            title = tag.strip()
            code = (f"# {title}\n" if title else "") + body
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return len(cells)


if __name__ == "__main__":
    print("cells:", convert(sys.argv[1], sys.argv[2]))
