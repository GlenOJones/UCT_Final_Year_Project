# uct-thesis-template

LaTeX template for a postgraduate thesis or dissertation in the Department of
Electrical Engineering, University of Cape Town. Used by the African Robotics
Unit (ARU).

Compiles with **pdfLaTeX**. Set in EB Garamond.

## Quick start

1. **Use this template** on GitHub, or clone.
2. Overleaf: **New Project → Import from GitHub**.
3. Fill in `0_documentproperties.tex`. **This is the only file you must edit
   before anything else.** Your name, student number, title, degree,
   supervisor, ORCID, ethics number.
4. Write your chapters into `1_intro.tex`, `2_review.tex` and so on.

```sh
latexmk 0_main
```

CI compiles the thesis on every push, so a broken `\include` or a missing
figure is caught immediately rather than the week before submission.

## Layout

### Front matter

| File | Contents |
|---|---|
| `0_documentproperties.tex` | **the file you edit.** Every name, number and title in the thesis |
| `0_titlepage.tex` | the cover |
| `i_declaration.tex` | plagiarism declaration |
| `ii_abstract.tex` | abstract |
| `iii_acknowledgments.tex` | acknowledgements |
| `iv_listofsymbols.tex` | list of symbols |
| `v_glossary.tex` | glossary entries |

### Body

| File | Contents |
|---|---|
| `1_intro.tex` | introduction |
| `2_review.tex` | literature review |
| `3_ch3.tex` – `5_ch5.tex` | your work. Rename these to what they actually are |
| `6_results.tex` | results |
| `7_discussion.tex` | discussion |
| `8_conclusions.tex` | conclusions |
| `9_furtherwork.tex` | further work |

### Appendices

| File | Contents |
|---|---|
| `A_gitrepo.tex` | code repository |
| `B_ethics.tex` | ethics approval |
| `C_topicdescription.tex` | topic description (commented out by default) |
| `D_ECSA.tex` | ECSA outcomes (commented out by default) |

`C` and `D` are commented out in `0_main.tex`. Uncomment the ones your
degree requires.

## Every name and number lives in one file

`0_documentproperties.tex` holds the title, subtitle, degree, degree
abbreviation, your name, student number, employee ID, email, ethics number,
ORCID, code repository, department, university, supervisor, co-supervisor, HOD
and keywords.

They then appear on the title page, in the declaration and in the appendices
without being typed twice. Change your title once and it changes everywhere.

## Word count and page count

The declaration page states both, **excluding the preamble, the appendices and
the bibliography**. Neither is typed by hand.

**Page count** is computed in LaTeX. `\label{body:start}` sits before the first
body chapter and `\label{body:end}` after the last, and `\bodypages` is the
difference. It needs two LaTeX passes.

**Word count** cannot be done in LaTeX: by the time LaTeX sees the text it has
already lost the distinction between a word and a macro argument. It is
therefore produced by `latexmkrc`, which runs `texcount` over the body chapters
and writes `wordcount.tex` for the document to read back. If `texcount` is
unavailable, a pure-perl fallback in `latexmkrc` does an approximate count.

**When you add a chapter, add it to `@BODY` in `latexmkrc`**, or it will not be
counted.

`latexmkrc` runs pdfLaTeX twice per pass on purpose. On the first pass the
labels do not exist and the page count comes out as `1`, which is `0 - 0 + 1`
and is plausible enough to go unnoticed. Running twice makes that impossible.

`wordcount.tex` is generated and is in `.gitignore`. Do not commit it.

## Citation style

One switch, near the top of `0_main.tex`:

```latex
\newif\ifIEEEcite
\IEEEcitefalse      % Harvard  (Shannon, 1948)   <- default
% \IEEEcitetrue     % IEEE     [1]
```

It drives both the in-text citations and the bibliography. Harvard is the
default, because the declaration page commits you to the Harvard convention.
Change both if you switch.

**Why the square brackets were appearing:** `natbib` loaded with **no options**
defaults to *numeric* citations, which is where `[1]` came from, regardless of
which `.bst` was in use. natbib has to be told `authoryear` explicitly. The
citation style and the bibliography style are configured separately and will
not agree on their own.

IEEE mode uses `IEEEtran.bst`, which ships with the `IEEEtran` package.

## Glossary

Glossary entries are defined in `v_glossary.tex` and cited with `\gls{}`. The
glossary needs `makeglossaries` to run between LaTeX passes; `latexmkrc`
handles that. On Overleaf it is automatic.

## Notes

- Do not commit the built PDF. `.gitignore` excludes `*.pdf`, but carries
  exceptions for `pdfs/` and `figs/`, because a PDF that is *input* to the
  build (an ethics approval scan, a vector figure) is a source asset and the
  build fails without it.
- Keep your chapter figures in `figs/chN/` so that two chapters cannot collide
  on a filename.
- `thesis.bib` is the bibliography. The style was previously `kluwer`, but
  `kluwer.bst` is not part of TeX Live and was not in the repository, so bibtex
  failed with `I couldn't open style file kluwer.bst` and no bibliography was
  produced. If your department supplies a `kluwer.bst`, commit it and use it in
  the `\ifIEEEcite` block in `0_main.tex`.
- The research group logos on the title page (ARU, RRSG, MARiS, MMRU, SOCCO,
  SEA) are commented out in `\bottomlogos` in `0_documentproperties.tex`.
  Uncomment the ones your thesis belongs to. The EEE logo is on.
