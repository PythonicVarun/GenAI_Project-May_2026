import logging
import re
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_git_command(args: List[str]) -> Optional[str]:
    """Helper to run a git command and return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_git_commit_url() -> Optional[str]:
    """Generates a GitHub commit URL for the current repository state."""
    # Check if we are in a git repository
    is_inside = run_git_command(["rev-parse", "--is-inside-work-tree"])
    if is_inside != "true":
        return None

    commit_hash = run_git_command(["rev-parse", "HEAD"])
    if not commit_hash:
        return None

    remote_url = run_git_command(["config", "--get", "remote.origin.url"])
    if not remote_url:
        return None

    # Clean up the remote url
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]

    # SSH format: git@github.com:owner/repo
    ssh_match = re.match(r"git@github\.com:(.+)", remote_url)
    if ssh_match:
        remote_url = f"https://github.com/{ssh_match.group(1)}"
    else:
        # SSH protocol format: ssh://git@github.com/owner/repo
        ssh_proto_match = re.match(r"ssh://git@github\.com/(.+)", remote_url)
        if ssh_proto_match:
            remote_url = f"https://github.com/{ssh_proto_match.group(1)}"

    if "github.com" in remote_url:
        return f"{remote_url}/commit/{commit_hash}"
    elif "gitlab.com" in remote_url:
        return f"{remote_url}/-/commit/{commit_hash}"

    return f"{remote_url} (commit: {commit_hash})"


def get_uncommitted_changes() -> List[str]:
    """Returns a list of uncommitted/dirty files using git status --porcelain."""
    is_inside = run_git_command(["rev-parse", "--is-inside-work-tree"])
    if is_inside != "true":
        return []

    status_output = run_git_command(["status", "--porcelain"])
    if not status_output:
        return []

    return [line.strip() for line in status_output.split("\n") if line.strip()]


def check_git_status_and_confirm() -> bool:
    """Check for uncommitted changes and request confirmation in interactive shells."""
    uncommitted = get_uncommitted_changes()
    if not uncommitted:
        return True

    logger.warning("Uncommitted changes detected in the repository:")
    for change in uncommitted:
        logger.warning("  %s", change)

    # Check if in an interactive terminal
    if sys.stdin.isatty():
        try:
            print("\nWARNING: There are uncommitted changes in the repository.")
            print(
                "Running now means your WandB run might be linked to a commit that "
                "does not match your active code."
            )
            response = (
                input("Do you want to proceed with the run anyway? [y/N]: ")
                .strip()
                .lower()
            )
            if response in ("y", "yes"):
                logger.info("Proceeding with the run as confirmed by user.")
                return True
            else:
                logger.error("Run aborted by user.")
                return False
        except KeyboardInterrupt:
            logger.error("Run aborted (KeyboardInterrupt).")
            return False
    else:
        logger.warning(
            "Running in non-interactive environment (no TTY). "
            "Proceeding automatically with warning..."
        )
        return True
