# Blog Contributing Guide for [MLflow](https://mlflow.org/blog)

Welcome to the MLflow community! We're excited that you're interested in contributing to our blog and are eager to highlight your tutorials, use cases,
and best practices for using MLflow. 

This guide is designed to provide you with all the information you need to write, submit, and contribute a [blog post](https://mlflow.org/blog).

## Why Contribute?

Contributing to the official MLflow blog allows you to share your knowledge, experiences, and insights related to leveraging MLflow for management,
evaluation, training, and deployment of GenAI, Deep Learning, and Traditional ML models.
Sharing your use cases, tips, and best practices leveraging MLflow is a great way to contribute to the community, enhance your writing
portfolio, and help other MLflow users by sharing your valuable knowledge and experience.

## Content That We're Looking For

We welcome submissions on a wide range of topics, including but not limited to:

* How-tos, technical guides and tutorials relating to the use of MLflow for GenAI, DL, or ML use cases
* Deep dives into specific features within MLflow
* Use Case studies relating to the use of MLflow
* Best practices of using complex or advanced features within MLflow
* Tips and tricks for leveraging MLflow for common tasks in the ML Development lifecycle
* Overview and introduction to major new to-be released features in MLflow
* Summaries and reports from MLflow meetups or summit events

> Note: If you suggest a topic not listed above, we'll be happy to discuss its applicability to the blog within your filed proposal.

## Submission Process

* **Idea Proposal**: Before writing, submit a brief outline of your idea via an MLflow Website Issue submission using the ["Blog Post"](https://github.com/mlflow/mlflow-website/issues) issue type. Ensure that all elements are filled in within the filed issue, paying careful attention to the legal acknowledgement to ensure that you don't spend time writing about something that will never be able to be published.

* **Discussion and Revision**: An MLflow maintainer or Developer Relations member may request further clarification to the content of your blog from within the filed issue or suggestions and direction on the overall content of the blog.

* **Approval**: Wait for approval from one of the MLflow maintainers prior to working on your blog (to save you time and frustration). Maintainers will have an open dialog with you regarding the scope, content, and chosen topics to ensure that the blog content aligns with our vision of enabling open source ML developers to build effective ML projects.

* **PR Submission**: Once the subject of your blog has been approved (with confirmation from an MLflow maintainer), write your blog post following the guidelines provided below. File a PR from your feature branch and select review from the maintainer that approved your proposal from the filed Issue.

* **Review Process**: An MLflow maintainer will review your PR and provide feedback, corrections, and general guidance on adjusting the content, pacing, writing style, and embedded content to ensure that the PR meets with the standards for the MLflow website. As corrections are made and changes are pushed, work with the reviewer to get your blog PR in a state that it can be merged to the main branch.

* **Publication**: After your PR is merged, your blog post will be released with the next minor release of MLflow.

## Writing Guidelines

### Length

Aim for a **rendered page length** between 3 and 8 pages, including code examples and images.

#### How to determine blog post length

When estimating the total length of your blog, ensure that you are evaluating the final rendered blog (built locally, using `npm`, abiding by the guidance in [the development guide](DEVELOPMENT_GUIDE.md)) in a browser window set at 100% zoom at a 1080p resolution.

### Tone

The MLflow blog aims for a professional yet approachable technical tone.

* Be clear, concise, and friendly. This is a technical blog for an open-source project, not a dissertation.
* Avoid the use of unexplained technical jargon, industry-specific terms, or buzzwords. If you need to use them, define them first.
* Ensure that MLflow-related feature mentions and technical topics are linked to appropriate sections of the MLflow documentation.

### Original Content

All submissions must be original content that you own the rights to and hasn't been published elsewhere.

### Permissions / Legal

By submitting a post, you agree to allowing us to edit the content for clarity, style, and format. If corrections need to be made post-publication, we will make them on your behalf. You'll be credited as the author of the post, even if a complete overhaul of the blog content is required.

If you need approval from your organization to publish content regarding their intellectual property, ensure that you provide a verification contact with your blog proposal that will allow us to confirm that we have permission to publish this content.

### Use of GenAI

While using GenAI tools to generate structural elements of technical guides is certainly useful (we use it too!), the actual text content of the blog **must** be written by you.

We will assess the content of submissions to determine if the content is generated via GenAI tools and will request that the content be rewritten if it is determined that it has been tool-generated (We don't have a headshot or LinkedIn profile for GPT-4, DBRX, or Mixtral, after all).

The use of GenAI generated images is totally permitted, however. Feel free to use those with impunity. We recommend that the blog title card is generated with the creative use of GenAI tooling (although custom contextually-relevant images or `.gif`s are acceptable too).

### Formatting

Our blogs are written in Markdown (`.md`) format with a specific header style in `mdx` format that the site framework uses for rendering specific elements. If in doubt, refer to existing blog posts for guidance on the form and structure required.

When creating sections within the blog, do not exceed an `h4` depth level to avoid distracting and hard-to-read section delineation within your blog.

Be liberal with providing links to relevant sections within the MLflow documentation and links to external projects that are showcased. When linking to a particular topic, only provide a link for the first mention of the reference.

### Images

Include images to illustrate key points, architectures, UI results of operations, and diagrams of complex topics. If you are including content that you did not create, ensure that the license for use of such an image is permissible and does not encroach upon others' copyright or trademark rights. Images must be in either `.png` or `.svg` format. We do not accept `.jpg` / `.jpeg`, `.bmp`, `.webp`, or low-resolution `.gif` images.

#### Use of .gif format for videos

If you are embedding a video within your blog (typically from a screen capture), we recommend using [ffmpeg](https://ffmpeg.org/) to convert video files (e.g., `.mp4`) to a high-resolution `.gif`. We do not allow embedding html5 video links on the site. 

`ffmpeg` can be installed via the brew cask as follows:

```bash
brew install ffmpeg
```

In order to generate a high quality `.gif` that minimizes artifacts and dithering issues is to use the following settings:

```bash
ffmpeg -i <my_movie>.mp4 -vf "fps=10,scale=2048:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" <my_output_gif>.gif
```

### Diagrams

If developing an architectural or process diagram that is intending to explain a complex topic, we prefer the use of either [Excalidraw](https://excalidraw.com/) or [draw.io](https://app.diagrams.net/) for consistency and thematic reasons. If you submit a hand-crafted diagram that was not made with either of these tools, **you will be asked to regenerate the diagram** using one of these tools.

> IMPORTANT: If using `Excalidraw` to generated diagrams (**recommended**), ensure that when exporting your diagram to `.png` format that you enable the `Embed scene` option. This allows for editing of images that have been generated using Excalidraw, dramatically reducing the amount of effort to update or change the rendered `.png` version of your diagram.

### Embedded Code Guidelines

Code examples should be provided in a complete and runnable state. If showing Python code, for instance, ensure that you are including `pip` install of requirements prior to the first code block, as well as full import statements so that the code within the blog **can be run without any external contextual references**.

If a data source is required to execute the examples, provide the means within the examples of downloading an open-source dataset that is easily accessible (as well as code that will fetch and locally store the data).

Do not inadvertantly commit code that contains sensitive information (e.g., OpenAI keys).

### Spelling and Grammar

Blogs on MLflow use American English spelling and grammar. Please be patient if we ask for grammatical spelling corrections to your post.

### Links

Include links to relevant resources, but avoid promotional content (either organizational or personal) or links to competing tools (we don't want to start a flame war).

When linking to MLflow documentation pages:

* Ensure that you are generating a direct link to the `latest` version of the docs (no relative links or documentation version links). For example: `[transformers](https://www.mlflow.org/docs/latest/llms/transformers/index.html)`
* Do not repeat links. If you're mentioning a link to, say, [transformers](https://www.mlflow.org/docs/latest/llms/transformers/index.html), only link to the first mention of the term.  

### Accessibility

Aim for accessible writing and formatting, particularly with font sizes in images and overall density within diagrams.

For posted images, include alt text so that screen readers can effectively describe the diagram, image, or video-converted `.gif`.

### First time Contributors

If it's your first time contributing to the MLflow Blog, ensure that you enter a unique entry for yourself (and co-authors, if applicable) within the [authors](website/blog/authors.yml) YAML file. We parse from this collection to populate the links to the authors (including a headshot) so that readers can communicate with you about your contribution. Please ensure that the entries are correct and render well when developing your generated blog locally.

Part of this rendering involves a headshot that is resolvable through a public accessible site. Most contributors choose to use their LinkedIn headshot photo.

## Thank You

We appreciate your interest in contributing to the MLflow blog! If you have any questions regarding the process, contact an MLflow maintainer within your blog proposal Issue.

## Maintainers

For reference, here is the list of current MLflow Maintainers:

| Name              | GitHub Handle   |
|-------------------|-----------------|
| Ben Wilson        | `@BenWilson2`   |
| Corey Zumar       | `@dbczumar`     |
| Daniel Lok        | `@daniellok-db` |
| Harutaka Kawamura | `@harupy`       |
| Serena Ruan       | `@serena-ruan`  |
| Weichen Xu        | `@WeichenXu123` |
| Yuki Watanabe     | `@B-Step62`     |
