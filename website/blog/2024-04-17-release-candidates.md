---
title: MLflow Release Candidates
tags: [mlflow]
slug: release-candidates
authors: [mlflow-maintainers]
thumbnail: img/blog/release-candidates.png
---

# Announcing MLflow Release Candidates

We are excited to announce the implementation of a release candidate process for MLflow!
The pace of feature development in MLflow is faster now than ever before and the core maintainer team has even more exciting things planned in the near future! However, with an increased velocity on major feature development comes with a risk of breaking things. As the maintainers of such a widely used project, we are cognizant of the disruptive nature of regressions and we strive to avoid them as much as we can. Aside from new feature development work, our primary goal is in ensuring the stability of production systems. While we do have the aspirational goal of moving fast(er), we certainly don't want to move fast and break things. With that goal in mind, we've decided to introduce a release candidate (RC) process. The RC process allows us to introduce new features and fixes in a controlled environment before they become part of the official release.

## How It Works

Starting from MLflow 2.13.0, new MLflow major and minor releases will be tagged as release candidates (e.g., `2.13.0rc0`) in PyPI two weeks before they are officially released.

The release candidate process involves several key stages:

- Pre-Release Announcement: We will announce upcoming features and improvements, providing our community with a roadmap of what to expect.
- Release Candidate Rollout: A release candidate version will be made available for testing, accompanied by detailed release notes outlining the changes.
- Community Testing and Feedback: We encourage our users to test the release candidate in their environments and share their feedback with us by filing issue reports on the MLflow Github repository. This feedback is invaluable for identifying issues and ensuring the final release aligns with user needs (i.e., we didn't break your workflows).
- Final Release: After incorporating feedback and making necessary adjustments, we will proceed with the final release. This version will include all updates tested in the RC phase, offering a polished and stable experience for all users.

This approach provides several benefits:

- Enhanced Stability: By rigorously testing release candidates, we can identify and address potential issues early, reducing the likelihood of disruptions in production environments.
- Community Feedback: The RC phase offers you, a member of the MLflow community, the opportunity to provide feedback on upcoming changes. This collaborative approach ensures that the final release aligns with the needs and expectations of our users.
- Gradual Adoption: Users can choose to experiment with new features in a release candidate without committing to a full upgrade. This flexibility supports cautious integration and thorough evaluation in various environments.

## Get Involved

Your participation is crucial to the success of this process. We invite you to join us in testing upcoming release candidates and sharing your insights. Together, we can ensure that MLflow continues to serve as a reliable foundation for your machine learning projects.
Stay tuned for announcements regarding our first release candidate. We look forward to your contributions and feedback as we take this important step toward a more stable and dependable MLflow.
