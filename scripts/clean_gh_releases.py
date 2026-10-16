import os
import time

from github import Github
import datetime

# Authentication is defined via github.Auth
from github import Auth

# using an access token
auth = Auth.Token(os.environ["GITHUB_TOKEN"])

thirty_days_ago = datetime.date.today() - datetime.timedelta(days=30)

g = Github(auth=auth)

# Assets are deleted purely by age, whether or not anything newer replaces them,
# so a release that can go 30 days without an upload gets emptied out. That rules
# out eudsl-python-extras: deploy_pip_page.yml needs at least one of its sdists.
TAGS = ["llvm", "eudsl", "mlir-python-bindings"]

repo = g.get_repo("llvm/eudsl")
total_deleted = 0
missed = []
for tag in TAGS:
    release = repo.get_release(tag)
    # Deleting while paging through the assets shifts later pages, so keep
    # sweeping until a pass deletes nothing.
    for _ in range(100):
        n_deleted = 0
        assets = release.get_assets()
        for ass in assets:
            if ass.name == "Payload.zip":
                continue
            if ass.created_at.date() < thirty_days_ago:
                print(ass.name)
                assert ass.delete_asset()
                n_deleted += 1

        total_deleted += n_deleted
        if n_deleted == 0:
            break
        time.sleep(1)
    else:
        missed.append(tag)

print(f"deleted {total_deleted} assets")
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"n_deleted={total_deleted}\n")

if missed:
    raise Exception(f"missed some in {missed}")
