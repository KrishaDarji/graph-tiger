import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ["data/processed/**", "data/tiger_5core/Beauty/**", "results/**"]
IGNORE = ["**/epoch_*_model.pth", "results/lightgcn*/last.pt", "results/lightgcn_768/best.pt",
          "results/tiger_full_pilot/**", "**/__pycache__/**"]

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

action, paths = sys.argv[1], sys.argv[2:]  # push|pull [path ...]
allow = [x for p in paths for x in (p.strip("/"), p.strip("/") + "/**")] or ALLOW
repo, token = os.environ["HF_REPO"], os.environ["HF_TOKEN"]
if action == "pull":
    snapshot_download(repo, local_dir=ROOT, token=token, allow_patterns=allow, ignore_patterns=IGNORE)
else:
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=ROOT, allow_patterns=allow, ignore_patterns=IGNORE)
