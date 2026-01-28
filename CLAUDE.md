# CLAUDE.md

This is the official MLflow website repository (https://mlflow.org/), built with Docusaurus, React, and TypeScript.

## Development Commands

All commands run from the `website/` directory:

```bash
yarn              # Install dependencies
yarn start        # Start dev server at http://localhost:3000
yarn build        # Build production site
yarn fmt          # Format code with Prettier (required before commits)
yarn lint         # Run all validation checks
yarn typecheck    # TypeScript type checking
```

## Project Structure

- `website/src/components/` - React components
- `website/src/pages/` - Main pages (landing, ambassadors, etc.)
- `website/blog/` - Blog posts (dated directories)
- `website/releases/` - Release notes
- `website/static/docs/` - MLflow documentation (copied from main repo during deploy)

## Key Guidelines

- Run `yarn fmt` before committing - CI will fail if formatting differs
- Blog images: PNG/SVG only (no JPG/WebP)
- Blog authors must be defined in `website/blog/authors.yml`
