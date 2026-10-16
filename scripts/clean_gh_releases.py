import os
import re
import sys
from collections import defaultdict

from github import Auth, Github, GithubException

# GitHub refuses uploads to a release that already has this many assets.
MAX_ASSETS = 1000

# The per-build version in an asset name: 20261016+fa1e20e8a (llvm,
# mlir-python-bindings) or 0.1.0.20261016.1529+5c59ae9 (eudsl-python-extras).
# Assets without one (Payload.zip, the fixed-version eudsl and mlir-python-mcp
# wheels) are replaced in place by every build, never accumulate, and are never
# deleted.
VERSION_RE = re.compile(r"\d{8}(?:\.\d+)?\+[0-9a-f]+")

TAGS = ["llvm", "eudsl", "mlir-python-bindings", "eudsl-python-extras", "mlir-python-mcp"]


def make_room(release):
    """Delete the oldest builds until one more build fits under MAX_ASSETS."""
    assets = list(release.get_assets())
    builds = defaultdict(list)
    slots = set()
    for ass in assets:
        m = VERSION_RE.search(ass.name)
        if m:
            builds[m.group()].append(ass)
            slots.add(VERSION_RE.sub("", ass.name))

    # A build uploads one asset per slot (package + platform/python tag), so this
    # is how many the next build adds. Each project has a different number.
    incoming = len(slots)
    n_assets = len(assets)
    print(
        f"{release.tag_name}: {n_assets} assets, {len(builds)} builds, "
        f"{incoming} assets per build"
    )

    n_deleted = 0
    oldest_first = sorted(builds.values(), key=lambda b: min(a.created_at for a in b))
    # Never delete the newest build; it may be the one being uploaded right now.
    for build in oldest_first[:-1]:
        if n_assets + incoming <= MAX_ASSETS:
            break
        for ass in build:
            print(f"  deleting {ass.name}")
            try:
                ass.delete_asset()
            except GithubException as e:
                # The release jobs of one build run in parallel and prune the
                # same oldest builds, so another job may have got here first.
                if e.status != 404:
                    raise
            n_assets -= 1
            n_deleted += 1
    return n_deleted


def main():
    tags = sys.argv[1:] or TAGS
    repo = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])).get_repo("llvm/eudsl")
    total_deleted = sum(make_room(repo.get_release(tag)) for tag in tags)

    print(f"deleted {total_deleted} assets")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"n_deleted={total_deleted}\n")


if __name__ == "__main__":
    main()
