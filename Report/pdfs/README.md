# pdfs/

PDFs that are **input** to the build: an ethics approval scan, a signed topic
description form, a vector figure exported from another tool.

They are source assets, not build artefacts. `.gitignore` excludes `*.pdf`
across the repo but carries an explicit exception for this directory, or the
build would fail for anyone who cloned it.

Do not put the built thesis here.
