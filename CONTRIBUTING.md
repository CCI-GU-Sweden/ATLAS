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

After documentation changes are merged, run **Documentation** on `main` and
select **Publish the main branch to GitHub Pages**. Open the URL reported by
the deployment and verify the pages and links. The repository's **Settings →
Pages** source must be set to **GitHub Actions**.

## Release the backend

The backend distribution is `atlas-cci-align`. Its version is derived from the
Git tag by setuptools-scm. Use an explicit `vX.Y.Z` tag for a stable release or
`vX.Y.ZrcN` for a release candidate. Before choosing a version, check existing
Git tags and PyPI releases. Use a new version for any changed source.

1. Ensure the `atlas-cci-align` PyPI Trusted Publisher is configured for
   `publish-backend-pypi.yml`.
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

## Release the Napari plugin

The plugin's version is derived from its Git tag, independently of the
backend. Use `napari-vX.Y.ZrcN` for a release candidate or `napari-vX.Y.Z`
for a stable release. Check existing tags and PyPI releases before choosing a
version, and update `plugin/CHANGELOG.md` for the release.

1. Update the changelog for the release, review it in a PR, and merge it to
   `main`. Configure the `napari-atlas-cci` PyPI Trusted
   Publisher for this repository and `publish-plugin-pypi.yml`.
2. Run **Build plugin release** on `main` without a tag to rehearse the
   build and widget discovery. The rehearsal does not publish to PyPI.
3. Once CI and the rehearsal pass, create an annotated tag pointing to the
   reviewed `main` commit. For example:

   ```sh
   git fetch origin --tags
   git tag -a napari-vX.Y.Z <commit-sha> -m "Release napari-atlas-cci X.Y.Z"
   git push origin refs/tags/napari-vX.Y.Z
   ```

4. Review the tagged **Build plugin release** run and its wheel and source
   archive. Save its run ID. The build checks the tag against the version and
   `main`, then verifies the installed widget in an offscreen Napari session.
5. Run **Publish plugin release to PyPI** on `main`, entering the run ID and
   confirming the public upload. It publishes the reviewed artifacts without
   rebuilding them. Install the exact candidate in a separate environment to
   verify the PyPI release.
