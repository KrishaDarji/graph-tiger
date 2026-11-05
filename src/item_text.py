"""Build item text from Amazon metadata for the Sentence-T5 encoder."""
import ast
import gzip
import html
import json


def read_metadata(path, wanted=None):
    """Return {asin: metadata row}, optionally limited to the wanted asins."""
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            asin = row.get("asin")
            if asin is not None and (wanted is None or asin in wanted):
                rows[asin] = row
    return rows


def _categories(row):
    seen = []
    for path in row.get("categories") or []:
        for name in path:
            if name not in seen:
                seen.append(name)
    return ", ".join(seen)


def project_text(row):
    """Title, brand, categories, description. salesRank is excluded: it is a
    popularity snapshot from metadata collection time and could leak the future.
    The description goes last so truncation only ever cuts description text."""
    fields = [("Title", row.get("title")), ("Brand", row.get("brand")),
              ("Categories", _categories(row)), ("Description", row.get("description"))]
    return "\n".join(f"{name}: {html.unescape(str(value))}" for name, value in fields if value)


def upstream_text(row):
    """The XiaoLongtaoo/TIGER template, used only to reproduce its 5-core result."""
    return (f"'title':{row.get('title') or None}\n 'price':{row.get('price') or None}\n "
            f"'salesRank':{row.get('salesRank') or None}\n 'brand':{row.get('brand') or None}\n "
            f"'categories':{row.get('categories') or None}")


TEMPLATES = {"project": project_text, "upstream": upstream_text}
