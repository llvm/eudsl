import os
import time

from github import Github
import datetime

# Authentication is defined via github.Auth
from github import Auth

# using an access token
auth = Auth.Token(os.environ["GITHUB_TOKEN"])

twelve_days_ago = datetime.date.today() - datetime.timedelta(days=12)

g = Github(auth=auth)

# gh api \
#   -H "Accept: application/vnd.github+json" \
#   -H "X-GitHub-Api-Version: 2022-11-28" \
#   /repos/llvm/eudsl/releases

n_deleted = 0
for rel in [
    # llvm
    191777437,
    # eudsl
    253847293,
    # mlir-python-bindings
    253847610
]:
    for _ in range(100):
        n_deleted = 0
        repo = g.get_repo("llvm/eudsl")
        release = repo.get_release(rel)
        assets = release.get_assets()
        for ass in assets:
            if ass.created_at.date() < twelve_days_ago:
                print(ass.name)
                assert ass.delete_asset()
                n_deleted += 1

        if n_deleted == 0:
            break
        time.sleep(1)

if n_deleted > 0:
    raise Exception("missed some")
