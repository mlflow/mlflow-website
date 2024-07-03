# Dev tools for the MLflow website

## Local generation of release notes formatting

If you would like to generate the required links for all release notes in the `website/releases/` directory, you can utilize the 
dev script located at `dev/local-format.js` as follows:

From the root of the repository, simply run:

```shell
node dev/local-format.js
```

If no files that require formatting are found, the script will report this. Otherwise, it will report a list of all files that have been 
updated with links.

For example, for a release note that hasn't had its links updated yet, you will see:

```shell
Updated files:
<path-to-repo>/mlflow-website/website/releases/2024-07-02-2.14.2-release.md
```

NOTE: The script will not update release notes prior to the MLflow 1.x release
