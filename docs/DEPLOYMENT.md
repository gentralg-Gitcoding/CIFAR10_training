# Documentation deployment guide

## Local preview

To build and preview the documentation locally:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme nbsphinx sphinx-autodoc-typehints ipykernel

# Build and serve
cd docs
./serve.sh
```

Then open http://localhost:8000 in your browser.

## GitHub Pages deployment

The documentation is automatically built and deployed to GitHub Pages when you push to the main branch.

### Initial setup

1. **Enable GitHub Pages in your repository:**
   - Go to Settings → Pages
   - Under "Build and deployment":
     - Source: GitHub Actions

2. **Configure Poetry extras (already done):**
   The `pyproject.toml` file includes documentation dependencies in the `docs` group.

3. **Workflow is configured:**
   The `.github/workflows/docs.yml` file handles automatic builds and deployment.

### Manual deployment

To manually trigger deployment:

1. Push to main branch:
   ```bash
   git add .
   git commit -m "Update documentation"
   git push origin main
   ```

2. Monitor the workflow:
   - Go to Actions tab in your GitHub repository
   - Check the "Deploy Documentation" workflow

3. Access your docs at:
   https://gperdrizet.github.io/CIFAR10

## Build warnings

The current build has the following expected warnings:

1. **Duplicate object descriptions**: Functions are documented in both quickstart and API reference. This is intentional for better UX.

2. **Display version theme option**: This warning is harmless and can be ignored.

## Updating documentation

### Adding new modules

1. Create a new `.rst` file in `docs/source/api/`
2. Add autodoc directives
3. Include in `docs/source/api/index.rst` toctree

### Adding new notebooks

1. Create the notebook in `notebooks/`
2. Add a link and description in `docs/source/notebooks/index.rst`

### Modifying configuration

Edit `docs/source/conf.py` to change:
- Theme settings
- Extensions
- API documentation behavior
- Intersphinx mappings

## Troubleshooting

### Build fails with import errors

Ensure the package is installed in editable mode:
```bash
pip install -e .
```

### GitHub Actions fails

Check that:
1. Repository has Pages enabled
2. Workflow has correct permissions in `docs.yml`
3. Dependencies are correctly specified in `pyproject.toml`
