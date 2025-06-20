module.exports = async ({ github, context }) => {
  const prNumber = parseInt(process.env.PR_NUMBER);
  const netlifyDomain = process.env.NETLIFY_DOMAIN;
  const deployActionUrl = process.env.DEPLOY_ACTION_URL;
  const buildActionUrl = process.env.BUILD_ACTION_URL;

  if (!prNumber) {
    console.log("No PR found for this workflow run");
    return;
  }

  const previewUrl = `https://pr-${prNumber}--${netlifyDomain}.netlify.app`;

  const comment = `🚀 **Netlify Preview Deployed!**

**Preview URL:** ${previewUrl}
**PR:** #${prNumber}
**Build Action:** ${buildActionUrl}
**Deploy Action:** ${deployActionUrl}

This preview will be updated automatically on new commits.`;

  // Find existing comment using pagination
  let botComment = null;

  for await (const response of github.paginate.iterator(
    github.rest.issues.listComments,
    {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNumber,
      per_page: 100,
    }
  )) {
    const found = response.data.find(
      (comment) =>
        comment.user.type === "Bot" &&
        comment.body.includes("Netlify Preview Deployed")
    );

    if (found) {
      botComment = found;
      break;
    }
  }

  if (botComment) {
    // Update existing comment
    await github.rest.issues.updateComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: botComment.id,
      body: comment,
    });
  } else {
    // Create new comment
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNumber,
      body: comment,
    });
  }
};
