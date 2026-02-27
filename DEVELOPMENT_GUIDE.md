# Development setup guide for MLflow Website contributions

This guide covers the general setup requirements that are needed to contribute to the MLflow website.

The core of the MLflow website is powered by [Docusaurus](https://docusaurus.io/).

## Fork this repo

In order to file a PR to this repository, you must file your PR from your fork of this repo. PRs that are submitted directly from the main branch will be closed.

## Setting up your local environment

In order to leverage local development tooling (including a rich live-preview capability that exists within the Docusaurus framework), you will
need to ensure that you have a compatible version of Node.js installed on your system.

### Node.js setup

We highly recommend using [Node Version Manager (nvm)](https://github.com/nvm-sh/nvm) to manage your Node.js environment. You can follow the instructions [here](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) to get `nvm` setup on your machine.

Once you have `nvm` installed, you will need to ensure that you have a Node.js version that is `v18` or later. You can follow the instructions in [this guide](https://nodesource.com/blog/installing-node-js-tutorial-using-nvm-on-mac-os-x-and-ubuntu/) to install and activate an appropriate version.

### npm setup

With the above steps completed, nvm will have acquired both node and `npm` (The Node package manager). npm is included with Node.js, so no additional installation is required.

## Building your development environment

To fetch and build the development environment for Docusaurus, simply navigate, from the root of your fork:

```bash
cd website
```

From this directory, run the following command:

```bash
npm install
```

This will fetch the required packages and ensure that your environment is up to date with required dependencies to build or start an interactive environment.

## Development

There are two primary activities that can be worked on in this repository:

- New page development or updates to existing main pages
- Blog writing

The recommended development process for validating changes for these two different workstreams are quite different.

### Core site development

When working on new pages or updating existing pages within the website, it can beneficial to use the live viewer mode for Docusaurus.
In this mode, `npm` will start a local server that will respond to the state of the files from your development branch, dynamically updating
a live view of the site within a browser window.

To enable live preview, simply run:

```bash
npm start
```

This will initialize the local development server, providing a local url that can be opened for live feedback as you make changes.

### Blog development

Example PRs for adding new blog posts: [#3](https://github.com/mlflow/mlflow-website/pull/3) [#39](https://github.com/mlflow/mlflow-website/pull/39)

Our blogs are written in markdown format. To test your changes after drafting a post, you can either start a local server with `npm start` or fully build the website as a static site.

To build the full static site content, you can run:

```bash
npm run build
```

### Linting

In order for your contribution to pass CI checks, the content of your PR must pass lint checks.
Prior to pushing your changes to your remote fork (and definitely prior to filing a PR from that feature branch), you must run
the lint formatter:

```bash
npm run fmt
```

This will automatically adjust your file contents to ensure that the linting rules are adhered to.

## Maintainer References

For maintainers, a reminder: this repository does not contain the documentation for MLflow. Changes to the docs are handled directly within the
[MLflow Repository](https://github.com/mlflow/mlflow/tree/master/docs/source) and are incorporated during release within the [docs](https://github.com/mlflow/mlflow-website/tree/main/website/static/docs) section of the repository prior to deployment.

To test inclusion of the current state of the MLflow documentation with a local build of this site, you can locally copy the generated static html
from the main repository:

> NOTE: Don't commit the contents of this directory to this repository!

```bash
cd /path/to/mlflow/docs
make rsthtml

mkdir -p /path/to/mlflow-website/website/static/docs/latest
cp -r build/html/* /path/to/mlflow-website/website/docs/latest
```

### Release tooling

If you are manually editing release notes or updating an existing release note, you can validate the generated links locally prior to pushing a PR by
running the script ``npm run fmt-notes`` to run local generation for all release notes that are within the ``releases`` directory.

### Preview

When a new commit is pushed to the `main` branch, the website is automatically built and deployed to the `gh-pages` branch. You can preview the website at the following URL:

https://mlflow.github.io/mlflow-website/

### Reference PRs

An example PR for adding a new page to the site: [#22](https://github.com/mlflow/mlflow-website/pull/22)

### Quick references

#### Requirements

- `node>=18`
- `npm`

#### Commands

```bash
cd website

# Install dependencies
npm install

# Start development server
npm start

# Build production website
npm run build

# Format code
npm run fmt

# Generate links within release notes
npm run fmt-notes
```
