#!/usr/bin/env bash
# Checks for broken cross-references and citations without running a build.
#   ./check_refs.sh            check the sources and the last VS Code build
#   ./check_refs.sh LABEL      list every use of LABEL before renaming/removing it
cd "$(dirname "$0")" || exit 1

if [ -n "$1" ]; then
    grep -n --include='*.tex' -r -- "$1" .
    exit
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Only the files 0_main.tex actually \include{}s, with comments stripped
files=$(grep -oE '^[^%]*\\(include|input)\{[^}]*\}' 0_main.tex | sed 's/.*{//;s/}//;s/$/.tex/')
for f in $files; do [ -f "$f" ] && sed 's/\([^\\]\)%.*/\1/;s/^%.*//' "$f"; done > "$tmp/all.tex"

grep -oE '\\label\{[^}]*\}' "$tmp/all.tex" | sed 's/.*{//;s/}//' | sort > "$tmp/labels"
grep -oE '\\([cC]|auto|eq|page|name)?ref\*?\{[^}]*\}|\\hyperref\[[^]]*\]' "$tmp/all.tex" \
    | sed 's/.*[{[]//;s/[]}]$//' | tr ',' '\n' | sed 's/ //g' | sort -u > "$tmp/refs"
grep -oE '\\(cite[a-zA-Z]*|citeurl|nocite)\*?(\[[^]]*\])*\{[^}]*\}' "$tmp/all.tex" \
    | sed 's/.*{//;s/}//' | tr ',' '\n' | sed 's/ //g' | grep -v '^#' | sort -u > "$tmp/cites"
cat thesis.bib FIXED.bib | grep -oE '^@[a-zA-Z]+\{[^,]*' | sed 's/.*{//' | sort -u > "$tmp/keys"

echo "== Refs with no matching \\label";       comm -23 "$tmp/refs" <(sort -u "$tmp/labels")
echo "== Labels defined more than once";       uniq -d "$tmp/labels"
echo "== Cites not in thesis.bib/FIXED.bib";   comm -23 "$tmp/cites" "$tmp/keys"
echo "== Last build log (0_main.log)";         grep -E 'undefined|multiply defined' 0_main.log | sort -u
echo "== BibTeX warnings (0_main.blg)";        grep -E 'Warning|error' 0_main.blg
