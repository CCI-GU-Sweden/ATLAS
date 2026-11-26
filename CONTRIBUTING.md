# Contributing to ATLAS

## Develop the backend

Run these commands from the repository root. If you already have the
`atlas-env` environment, reinstall the editable package so it points at
the backend's current location.

```sh
conda activate atlas-env
python -m pip install -e "./backend[test]"
python -m pytest backend/tests backend/examples/tests
```

## Backend distributions

The wheel ships the `atlas` package and its license. The source distribution
also includes the backend README, changelog, tests, and build configuration.
Notebooks and standalone scripts live in `backend/examples/`. Its separate
tests run in CI but are kept with the examples in Git.

## Preview and publish the documentation

From the ATLAS repository root, activate an existing `docs-env` with MkDocs
Material, or create it with `conda env create --file environment-docs.yml`.

```sh
conda activate docs-env
mkdocs build --strict
mkdocs serve
```

Open `http://127.0.0.1:8000/ATLAS/` to preview the site. The **Documentation**
workflow also builds the site on GitHub and lets reviewers download the
generated HTML as an artifact.

For the first deployment, select **GitHub Actions** as the source in the
ATLAS repository's **Settings → Pages**. After the docs PR is merged and
reviewed, run **Documentation** on `main` and select **Publish the main branch
to GitHub Pages**. Open the URL reported by the deployment and verify the home
page, the backend pages, and their links.

## Release the backend

The backend distribution is `atlas-cci-align`. Its version is derived from the
Git tag by setuptools-scm. Use an explicit `vX.Y.Z` tag for a stable release or
`vX.Y.ZrcN` for a release candidate. Before choosing a version, check existing
Git tags and PyPI releases. Use a new version for any changed source.

1. Before merging the release workflow transition, disable the legacy
   **Release** workflow in GitHub Actions. Leave **CI** enabled. On PyPI,
   configure the ATLAS Trusted Publisher to use
   `publish-backend-pypi.yml` and remove the legacy `release.yml` publisher
   once the new setup is ready.
2. In GitHub Actions, use **Run workflow** on **Build backend release** to
   rehearse the build without creating a tag or publishing anything.
3. For a release, select a reviewed commit on `main` with passing CI. From a
   clean clone with current branches and tags, create and push an annotated
   tag. Replace the placeholders below with the chosen version and commit.

   ```sh
   git fetch origin --tags
   git tag -a vX.Y.Z <commit-sha> -m "Release atlas-cci-align X.Y.Z"
   git push origin refs/tags/vX.Y.Z
   ```

4. Review the resulting **Build backend release** run. It checks the tag
   against `main`, builds and validates the wheel and source archive, and
   smoke-tests the installed wheel outside the checkout. Save the run ID
   from its URL.
5. To publish, run **Publish backend release to PyPI** from `main`. Enter
   that run ID and check the publication confirmation. The workflow verifies
   that the tag still points to the build commit and uploads the two checked
   artifacts from that run.
