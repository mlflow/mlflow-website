# CLAUDE.md

This is the official MLflow website repository (https://mlflow.org/), built with Docusaurus, React, and TypeScript.

## Development Commands

All commands run from the `website/` directory:

```bash
npm install       # Install dependencies
npm start         # Start dev server at http://localhost:3000
npm run build     # Build production site
npm run fmt       # Format code with Prettier (required before commits)
npm run lint      # Run all validation checks
npm run typecheck # TypeScript type checking
```

## Project Structure

- `website/src/components/` - React components
- `website/src/pages/` - Main pages (landing, ambassadors, etc.)
- `website/blog/` - Blog posts (dated directories)
- `website/releases/` - Release notes
- `website/static/docs/` - MLflow documentation (copied from main repo during deploy)

## Key Guidelines

- Run `npm run fmt` before committing - CI will fail if formatting differs
- Blog images: PNG/SVG only (no JPG/WebP)
- Blog authors must be defined in `website/blog/authors.yml`

## SEO Page Guidelines (learned from PR #501 reviewer feedback)

When creating new SEO pages under `website/src/pages/`:

### File & URL Structure
- Pages go at root of `src/pages/` (e.g., `llm-evaluation.tsx`), NOT in subdirectories
- Import paths from root pages use `../components/`, not `../../components/`
- Internal links use root paths (e.g., `/llm-tracing`), never `/faq/` prefixed paths

### SEO & Naming
- Lead with the higher-ranked search term in URL, title, and content ordering
- SEO title pattern: `"[Primary Term] & [Secondary Term] ... | MLflow AI Platform"`
- Canonical URL: `mlflow.org/<page-slug>`

### Terminology & Style
- "built-in" not "pre-built" (matches MLflow docs)
- "agents" or "agents and LLM applications" not "AI systems" or "AI applications"
- Reference current/relevant frameworks — avoid outdated ones
- No em dashes — use commas, colons, periods, or parentheses instead

### Content Structure
- General concept questions first, MLflow-specific questions after
- MLflow-specific FAQ questions must explicitly include "MLflow" in the question text
- Verify all doc links against the current docs URL structure (e.g., paths under `genai/eval-monitor/` not old paths like `genai/llm-evaluate/`)

### Cross-Page Requirements
- Add backlinks to the new page from related existing pages (home features, ai-gateway, ai-observability, llm-tracing, etc.)
- Use consistent MLflow product definition: "MLflow is the largest open-source AI engineering platform"
- Info-box: "Thousands of organizations use MLflow to debug, evaluate, monitor, and optimize production-quality AI agents and LLM applications while controlling costs and managing access to models and data."
- Include: Linux Foundation backing, Apache 2.0 license, 30M+ monthly downloads, no vendor lock-in
