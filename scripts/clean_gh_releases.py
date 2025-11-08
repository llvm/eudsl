import os
import re
import time

from github import Github
import datetime

# Authentication is defined via github.Auth
from github import Auth

# using an access token
auth = Auth.Token(os.environ["GITHUB_TOKEN"])

thirty_days_ago = datetime.date.today() - datetime.timedelta(days=30)

g = Github(auth=auth)

# gh api \
#   -H "Accept: application/vnd.github+json" \
#   -H "X-GitHub-Api-Version: 2022-11-28" \
#   /repos/llvm/eudsl/releases

skip = [
    # last commit before typed-pointers in MLIR is removed
    "35ca64989"
]

skip_re = re.compile("|".join(skip))

n_deleted = 0
for rel in [
    # llvm
    253845177,
    # eudsl
    253847293,
    # mlir-python-bindings
    253847610,
]:
    for _ in range(100):
        n_deleted = 0
        repo = g.get_repo("llvm/eudsl")
        release = repo.get_release(rel)
        assets = release.get_assets()
        for ass in assets:
            if skip_re.search(ass.name):
                continue
            if ass.created_at.date() < thirty_days_ago:
                print(ass.name)
                assert ass.delete_asset()
                n_deleted += 1

        if n_deleted == 0:
            break
        time.sleep(1)

if n_deleted > 0:
    raise Exception("missed some")
