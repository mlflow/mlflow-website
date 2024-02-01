# MLflow Website

This repository contains the source code for [the MLflow website](https://mlflow.org/), which is built using [Docusaurus](https://docusaurus.io/).

## Preview

When a new commit is pushed to the `main` branch, the website is automatically built and deployed to the `gh-pages` branch. You can preview the website at the following URL:

https://mlflow.github.io/mlflow-website/

## Development

### Requirements

- `node>=18`
- `yarn`

### Commands

```bash
cd website

# Install dependencies
yarn

# Start development server
yarn start

# Build production website
yarn build

# Format code
yarn fmt

# Convert blog post and release notes into typescript code
yarn compile
```

### Building docs

This repository doesn't contain the MLflow documentation. To populate the `/website/static/docs` directory, run the following commands:

```bash
cd /path/to/mlflow/docs
make rsthtml

cp -r build/html/* /path/to/mlflow-website/website/docs
```
