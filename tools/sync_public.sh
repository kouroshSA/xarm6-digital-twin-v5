#!/usr/bin/env bash
# Publish the private dev repo's main to the public repo, with Claude tooling
# metadata stripped from commit messages.
#
#   tools/sync_public.sh            # dry run: filter, verify, show what would push
#   tools/sync_public.sh --push     # push (fast-forward only)
#   tools/sync_public.sh --push --force   # first run after a policy change only
#
# Stripped: "Claude-Session: <url>" lines and "Co-Authored-By: ... <noreply@
# anthropic.com>" trailers. Trees, authors and dates are untouched, so the
# public tree at every commit is byte-identical to dev's.
#
# The rewrite is DETERMINISTIC: the whole history is re-filtered every run, and
# identical input gives identical hashes. So the public side stays a series of
# fast-forwards, and --force is needed only when the filter rule itself
# changes. dev is never modified; it keeps the full original messages.
#
# Remotes expected: `dev` (private, source) and `origin` (public, target).
set -euo pipefail

SRC_REMOTE=dev
DST_REMOTE=origin
BRANCH=main
PUSH=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git -C "$ROOT" fetch -q "$SRC_REMOTE"
git -C "$ROOT" fetch -q "$DST_REMOTE"
SRC_TIP=$(git -C "$ROOT" rev-parse "$SRC_REMOTE/$BRANCH")
DST_TIP=$(git -C "$ROOT" rev-parse "$DST_REMOTE/$BRANCH")

# Filter in a throwaway BARE repo, so no ref in the working clone is ever
# rewritten (and HEAD resolves: filter-branch dies on an unborn HEAD).
git init -q --bare --initial-branch="$BRANCH" "$WORK/repo"
cd "$WORK/repo"
git fetch -q "$ROOT" "$SRC_TIP:refs/heads/$BRANCH"
git fetch -q "$ROOT" "$DST_TIP:refs/remotes/public/$BRANCH"

# Drop the lines, then any blank lines they leave at the end of the message.
# A message with nothing to strip passes through byte-for-byte: otherwise awk
# appends a newline to the GitHub web-merge messages that lack one, and their
# hashes change for no reason.
cat > "$WORK/msg_filter.sh" <<'FILTER'
#!/bin/sh
PAT='^(Claude-Session: |Co-Authored-By: .*<[Nn][Oo][Rr][Ee][Pp][Ll][Yy]@[Aa][Nn][Tt][Hh][Rr][Oo][Pp][Ii][Cc]\.[Cc][Oo][Mm]>[[:space:]]*$)'
msg=$(mktemp); cat > "$msg"
if ! grep -qE "$PAT" "$msg"; then cat "$msg"; rm -f "$msg"; exit 0; fi
grep -vE "$PAT" "$msg" |
  awk '{ l[NR] = $0 }
       END { n = NR; while (n > 0 && l[n] ~ /^[[:space:]]*$/) n--
             for (i = 1; i <= n; i++) print l[i] }'
rm -f "$msg"
FILTER
chmod +x "$WORK/msg_filter.sh"
if ! FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
      --msg-filter "$WORK/msg_filter.sh" "$BRANCH" >"$WORK/fb.log" 2>&1; then
  cat "$WORK/fb.log" >&2; echo "FAIL: filter-branch" >&2; exit 1
fi

NEW_TIP=$(git rev-parse "$BRANCH")

# --- verify before anything leaves this machine ----------------------------
# Trailer lines as the filter defines them, plus a session URL ANYWHERE -- the
# URL is the thing that must not leak; prose that merely names the trailer
# (like this script's own commit message) is fine.
LEFT=$(git log "$BRANCH" --format=%B |
       grep -ciE '^Claude-Session: |^Co-Authored-By: .*<noreply@anthropic\.com>|claude\.ai/code/session' || true)
if [ "$LEFT" != "0" ]; then
  echo "FAIL: $LEFT stripped-pattern lines survived the filter" >&2; exit 1
fi
if [ "$(git rev-parse "$BRANCH^{tree}")" != "$(git -C "$ROOT" rev-parse "$SRC_TIP^{tree}")" ]; then
  echo "FAIL: filtered tip tree differs from $SRC_REMOTE/$BRANCH" >&2; exit 1
fi
if [ "$(git rev-list --count "$BRANCH")" != "$(git -C "$ROOT" rev-list --count "$SRC_TIP")" ]; then
  echo "FAIL: commit count changed" >&2; exit 1
fi

if git merge-base --is-ancestor "public/$BRANCH" "$BRANCH"; then
  KIND="fast-forward ($(git rev-list --count "public/$BRANCH..$BRANCH") new commits)"
  NEED_FORCE=0
else
  KIND="HISTORY REWRITE (public/$BRANCH is not an ancestor)"
  NEED_FORCE=1
fi
echo "source   $SRC_REMOTE/$BRANCH  ${SRC_TIP:0:7}"
echo "public   $DST_REMOTE/$BRANCH  ${DST_TIP:0:7}"
echo "filtered                ${NEW_TIP:0:7}   $KIND"

if [ "$NEW_TIP" = "$DST_TIP" ]; then echo "public is already up to date."; exit 0; fi
if [ "$PUSH" = 0 ]; then echo "dry run; pass --push to publish."; exit 0; fi
if [ "$NEED_FORCE" = 1 ] && [ "$FORCE" = 0 ]; then
  echo "refusing: this would rewrite public history. Re-run with --push --force if intended." >&2
  exit 3
fi

DST_URL=$(git -C "$ROOT" remote get-url "$DST_REMOTE")
git push --force-with-lease="$BRANCH:$DST_TIP" "$DST_URL" "$NEW_TIP:refs/heads/$BRANCH"
git -C "$ROOT" fetch -q "$DST_REMOTE"
echo "published ${NEW_TIP:0:7} to $DST_REMOTE/$BRANCH"
