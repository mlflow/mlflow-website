module.exports = async ({ github, context }) => {
  const runId = context.payload.workflow_run.id;
  const artifactName = process.env.ARTIFACT_NAME;
  const extractPath = process.env.EXTRACT_PATH || ".";

  const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
    owner: context.repo.owner,
    repo: context.repo.repo,
    run_id: runId,
  });

  const matchArtifact = artifacts.data.artifacts.find((artifact) => {
    return artifact.name === artifactName;
  });

  if (!matchArtifact) {
    throw new Error(`Artifact ${artifactName} not found`);
  }

  const download = await github.rest.actions.downloadArtifact({
    owner: context.repo.owner,
    repo: context.repo.repo,
    artifact_id: matchArtifact.id,
    archive_format: "zip",
  });

  const fs = require("fs");
  const { execSync } = require("child_process");

  // Write the zip file
  const zipPath = `${artifactName}.zip`;
  fs.writeFileSync(zipPath, Buffer.from(download.data));

  // Extract to the specified path
  execSync(`unzip -o ${zipPath} -d ${extractPath}`);

  // Clean up zip file
  fs.unlinkSync(zipPath);

  console.log(`Artifact ${artifactName} extracted to ${extractPath}`);
};
