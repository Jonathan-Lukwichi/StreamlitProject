#!/usr/bin/env python3
"""Auto-commit hook for Claude Code - commits and pushes after file edits."""
import json
import sys
import subprocess
import os
from datetime import datetime


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # Only process Edit or Write operations
    if tool_name not in ("Edit", "Write") or not file_path:
        sys.exit(0)

    # Get project directory
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    os.chdir(project_dir)

    # Get relative path for commit message
    try:
        rel_path = os.path.relpath(file_path, project_dir)
    except ValueError:
        rel_path = os.path.basename(file_path)

    # Stage the file
    subprocess.run(["git", "add", file_path], capture_output=True)

    # Check if there are staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )

    if result.returncode == 0:
        sys.exit(0)  # Nothing to commit

    # Create commit message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"Update: {rel_path} [{timestamp}]"

    # Commit
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        capture_output=True
    )

    # Push to origin main
    subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
