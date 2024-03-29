# Development setup guide for MLflow Website contributions

This guide covers the general setup requirements that are needed to contribute to the MLflow website.

The core of the MLflow website is powered by [Docusaurus](https://docusaurus.io/), a REACT-based static website generator framework.

## Fork this repo

In order to file a PR to this repository, you must file your PR from your fork of this repo. PRs that are submitted directly from the main branch will be closed.

## Setting up your local environment

In order to leverage local development tooling (including a rich live-preview capability that exists within the Docusaurus framework), you will
need to ensure that you have a compatible version of Node.js installed on your system.

### Node.js setup

The easiest way to get started with Node is to use the [Node Version Manager (nvm)](https://github.com/nvm-sh/nvm). `nvm` will allow you to seamlessly manage
different deployments of Node.js, offering simple command line tool access to switch between versions.

> Note: on OSX, you will need to have the Xcode command line tools installed before installing `nvm`. These can be directly installed from [Apple Developer Resources](https://developer.apple.com/xcode/resources/).

Installation of `nvm` can be done directly via a `curl` (alternatively, you can use `wget`) command:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

Once nvm is installed, you can download and install a version of node that is supported with Docusaurus.

```bash
nvm ls
```

An example output of running the listing command is:

```bash
➜ nvm ls
->     v16.20.2
       v20.11.0
         system
default -> 16 (-> v16.20.2)
iojs -> N/A (default)
unstable -> N/A (default)
node -> stable (-> v20.11.0) (default)
stable -> 20.11 (-> v20.11.0) (default)
lts/* -> lts/iron (-> v20.11.0)
lts/argon -> v4.9.1 (-> N/A)
lts/boron -> v6.17.1 (-> N/A)
lts/carbon -> v8.17.0 (-> N/A)
lts/dubnium -> v10.24.1 (-> N/A)
lts/erbium -> v12.22.12 (-> N/A)
lts/fermium -> v14.21.3 (-> N/A)
lts/gallium -> v16.20.2
lts/hydrogen -> v18.19.0 (-> N/A)
lts/iron -> v20.11.0
```

In order to download a version of node using nvm that meets the `node>=18` requirements, you can:

```bash
nvm install lts/iron
```
This will fetch, per the list above, node `v20.12.0`.

```bash
➜ nvm install lts/iron
Downloading and installing node v20.12.0...
Downloading https://nodejs.org/dist/v20.12.0/node-v20.12.0-darwin-arm64.tar.xz...
############################################################################################################################################################ 100.0%
Computing checksum with sha256sum
Checksums matched!
Now using node v20.12.0 (npm v10.5.0)
(mlflow-dev-env)
```

Once you have node installed, you can activate this new version by running:

```bash
➜ nvm use v20.12.0
Now using node v20.12.0 (npm v10.5.0)
```

### Yarn setup

With the above steps completed, nvm will have acquired both node and `npm` (The Node package manager).

We'll use `npm` to install `yarn` so that we can do active development, build a static version of the website, and compile the blog markdown content into renderable `.tsx` files.

```bash
npm install --global yarn
```

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

This will initialize the local development server:

```bash
➜ yarn start
yarn run v1.22.19
$ docusaurus start

    ---------------------------------------------------------------------------------------------------------------------------

                                                  Update available 3.0.1 → 3.1.1

                        To upgrade Docusaurus packages with the latest version, run the following command:
        `yarn upgrade @docusaurus/core@latest @docusaurus/plugin-client-redirects@latest @docusaurus/preset-classic@latest
                   @docusaurus/module-type-aliases@latest @docusaurus/tsconfig@latest @docusaurus/types@latest`

    ---------------------------------------------------------------------------------------------------------------------------

[INFO] Starting the development server...
[SUCCESS] Docusaurus website is running at: http://localhost:3000/

✔ Client
  Compiled successfully in 7.20s

client (webpack 5.89.0) compiled successfully
```

### Blog development

Example PRs for adding new blog posts: [#3](https://github.com/mlflow/mlflow-website/pull/3) [#39](https://github.com/mlflow/mlflow-website/pull/39)

Our blogs are written in markdown format, which is compiled to typescript files within Docusaurus for site display. Due to this
transpile stage, in order to validate the formatting and structure of your blog (and the correct rendering of images or other)
embedded content, you will have to compile your work in progress before building the site locally for verification.

To compile the blog content into the required `.tsx` files, you can run:

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
```