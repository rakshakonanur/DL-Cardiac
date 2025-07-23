import os
from pathlib import Path
import subprocess

"""
    Updates all git submodules in the specified directory.
"""

current_dir = Path.cwd()
base_dir = current_dir.parent.parent

for name in os.listdir(base_dir):
    repo_path = os.path.join(base_dir, name)
    print(f"Checking {name} at {repo_path}")
    if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"🔄 Updating {name}")
        subprocess.run(["git", "-C", repo_path, "pull"])

