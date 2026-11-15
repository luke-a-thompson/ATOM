#!/usr/bin/env bash
# Sync the Overleaf paper (git subtree at paper/) with this repo.
#
#   scripts/paper_sync.sh pull   # bring Overleaf edits into paper/
#   scripts/paper_sync.sh push   # send paper/ commits back to Overleaf
#
# Auth: the Overleaf git username is "git" and the password is an Overleaf
# token (olp_...). Never commit the token. Either
#   - run in an environment whose network proxy injects it (cloud sessions), or
#   - export OVERLEAF_TOKEN=olp_... before running, or
#   - let git's credential helper prompt / remember it.
set -euo pipefail

PROJECT_ID="67582be27ea98088ba09e7c6"
PREFIX="paper"
BRANCH="main"
REMOTE_URL="https://git@git.overleaf.com/${PROJECT_ID}"

cd "$(git rev-parse --show-toplevel)"

url="$REMOTE_URL"
if [[ -n "${OVERLEAF_TOKEN:-}" ]]; then
  url="https://git:${OVERLEAF_TOKEN}@git.overleaf.com/${PROJECT_ID}"
fi

if ! git remote get-url overleaf >/dev/null 2>&1; then
  git remote add overleaf "$REMOTE_URL"
fi

case "${1:-}" in
  pull)
    git subtree pull --prefix="$PREFIX" "$url" "$BRANCH" \
      -m "Sync paper/ from Overleaf"
    ;;
  push)
    git subtree push --prefix="$PREFIX" "$url" "$BRANCH"
    ;;
  *)
    echo "usage: $0 {pull|push}" >&2
    exit 2
    ;;
esac
