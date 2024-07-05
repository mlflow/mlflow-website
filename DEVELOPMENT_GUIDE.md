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

### Yarn setup

With the above steps completed, nvm will have acquired both node and `npm` (The Node package manager). To install `yarn`, follow either [this guide](https://yarnpkg.com/getting-started/install) or the legacy npm-based [guide here](https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable) to ensure that you have yarn installed on your system.

## Building your development environment

To fetch and build the development environment for Docusaurus, simply navigate, from the root of your fork:

```bash
cd website
```

From this directory, run the following command:

```bash
yarn
```

This will fetch the required packages and ensure that your environment is up to date with required dependencies to build, compile, or start an interactive environment.

## Development

There are two primary activities that can be worked on in this repository:

- New page development or updates to existing main pages
- Blog writing

The recommended development process for validating changes for these two different workstreams are quite different.

### Core site development

When working on new pages or updating existing pages within the website, it can beneficial to use the live viewer mode for Docusaurus.
In this mode, `yarn` will start a local server that will respond to the state of the files from your development branch, dynamically updating
a live view of the site within a browser window.

To enable live preview, simply run:

```bash
yarn start
```

This will initialize the local development server, providing a local url that can be opened for live feedback as you make changes.

### Blog development

Example PRs for adding new blog posts: [#3](https://github.com/mlflow/mlflow-website/pull/3) [#39](https://github.com/mlflow/mlflow-website/pull/39)

Our blogs are written in markdown format, which is compiled to typescript files within Docusaurus for site display. Due to this
transpile stage, in order to validate the formatting and structure of your blog (and the correct rendering of images or other)
embedded content, you will have to compile your work in progress before building the site locally for verification.

To compile the blog content into the required `.tsx` files (for linking your blog to the navigation components within the site), you can run:

```bash
yarn compile
```

> Note: prior to committing any changes from your blog feature, the blog contents must be compiled. The CI system will not compile your
markdown into `.tsx` for you.

After the compilation finishes, you can either start a local server with `yarn start` or fully build the website as a static site.

To build the full static site content, you can run:

```bash
yarn build
```

### Linting

In order for your contribution to pass CI checks, the content of your PR must pass lint checks.
Prior to pushing your changes to your remote fork (and definitely prior to filing a PR from that feature branch), you must run
the lint formatter:

```bash
yarn fmt
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

If you are adding a new blog entry, ensure that you have compiled the indexing file by running, from the ``website`` directory, the ``yarn compile`` script.
This ensures that the main page listing will be updated with the most recent entry.

If you are manually editing release notes or updating an existing release note, you can validate the generated links locally prior to pushing a PR by
running the script ``yarn fmt-notes`` to run local generation for all release notes that are within the ``releases`` directory.

### Preview

When a new commit is pushed to the `main` branch, the website is automatically built and deployed to the `gh-pages` branch. You can preview the website at the following URL:

https://mlflow.github.io/mlflow-website/

### Reference PRs

An example PR for adding a new page to the site: [#22](https://github.com/mlflow/mlflow-website/pull/22)

### Quick references

#### Requirements

- `node>=18`
- `yarn`

#### Commands

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

# Generate links within release notes
yarn fmt-notes
```
