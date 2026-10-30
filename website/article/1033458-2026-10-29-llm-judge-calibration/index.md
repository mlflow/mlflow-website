---
title: "LLM Judge Calibration: Measure, Correct and Report"
description: "Learn how to calibrate an LLM judge: build a human-labeled set, measure agreement with kappa, correct bias, and report results with honest confidence intervals."
slug: llm-judge-calibration
tags: [llm judge calibration]
date: 2026-10-29
image: https://media.babylovegrowth.ai/blog-images/organization-30814/1793256713755_Reviewer-examining-an-agent-evaluation-trace.jpeg
---

![Reviewer examining an agent evaluation trace](https://media.babylovegrowth.ai/blog-images/organization-30814/1793256713755_Reviewer-examining-an-agent-evaluation-trace.jpeg)

Calibrating an LLM judge means checking its verdicts against human labels, measuring how far they diverge, and then either fixing the judge or correcting the scores it produces. You do it in four moves: label a representative set with experts, measure chance-corrected agreement, fix the rubric and known biases, and report results with confidence intervals that include the judge's own error. This guide walks through each move, with the numbers and papers behind them.

## TL;DR

- An uncalibrated judge gives biased scores. [Imperfect sensitivity and specificity skew naive evaluation results](https://proceedings.mlr.press/v306/lee26bl.html), and a human-labeled calibration set lets you correct for it.
- Measure agreement with Cohen's kappa (two raters) or Fleiss' kappa (three or more), not raw accuracy alone. Measure human-human agreement first, because a judge cannot be more aligned than its labelers.
- Test for position, verbosity and fluency bias separately. High run-to-run consistency does not prove the judge is valid.
- Report bias-corrected scores with intervals that carry uncertainty from both the test set and the calibration set.
- Re-run calibration whenever the judge model, the prompt or the task changes.

## Table of Contents

- [What calibrating an LLM judge actually means](#what-calibrating-an-llm-judge-actually-means)
- [Step 1: Build the human-labeled calibration set](#step-1-build-the-human-labeled-calibration-set)
- [Step 2: Measure agreement the right way](#step-2-measure-agreement-the-right-way)
- [Step 3: Detect and reduce judge bias](#step-3-detect-and-reduce-judge-bias)
- [Step 4: Calibrate the judge or correct its scores](#step-4-calibrate-the-judge-or-correct-its-scores)
- [Step 5: Report calibrated results honestly](#step-5-report-calibrated-results-honestly)
- [Step 6: Keep calibration current](#step-6-keep-calibration-current)
- [Our perspective: judges are software you have to test](#our-perspective-judges-are-software-you-have-to-test)
- [Calibrate judges on your own traces with MLflow](#calibrate-judges-on-your-own-traces-with-mlflow)
- [FAQ](#faq)
- [Sources](#sources)

## What calibrating an LLM judge actually means

An LLM-as-a-judge is a classifier. It reads an output and returns a label: pass or fail, a 1 to 5 score, or a preference between two answers. Like any classifier, it has a true positive rate and a false positive rate. If it passes 90% of your outputs, that does not mean 90% are good. It means the judge said so, and the gap between those two statements is what calibration measures.

The term covers two related jobs:

1. **Alignment**: changing the judge (rubric, examples, model, optimizer) so its verdicts match what trusted humans decide.
2. **Statistical correction**: leaving the judge as is, estimating its error rates from labeled data, and adjusting the reported metric and its uncertainty.

You usually need both. Alignment raises the ceiling, and correction keeps your reported numbers honest about what remains.

The reason this matters is practical. Teams use judges to gate releases, compare prompts and track regressions in production. A judge with a systematic lean can make a worse prompt look better, and nobody notices because the dashboard looks precise.

## Step 1: Build the human-labeled calibration set

Everything downstream depends on this set, so build it deliberately.

**Sample representatively.** Pull cases from real traffic or realistic test inputs, including the hard and ambiguous ones. A set made only of obvious passes and failures will make any judge look good.

**Use experts, not whoever is free.** [Chrome's guidance on building an expert judge](https://developer.chrome.com/docs/ai/evals/expert-judge) recommends expert-labeled alignment data, a fresh holdout kept apart from the data you tune on, and measuring human-human agreement. If your two best reviewers agree only 70% of the time, your rubric is the problem, and no judge prompt will fix it.

**Split it.** Keep one portion for developing the judge (examples, rubric edits) and one holdout you never touch until you measure. Tuning on your test labels gives you an optimistic number that fails in production.

**Label in context.** For agents, the thing being judged is usually a whole trajectory, not a final string. Reviewers need the same traces the judge sees: tool calls, retrieved context, intermediate reasoning. Otherwise you compare two different pieces of evidence.

### How big should the set be?

There is no universal number, and anyone quoting one is simplifying. The size drives the width of your uncertainty. One practitioner analysis of [calibration set sizing](https://www.google.com/search?q=How+big+does+your+LLM-as-judge+calibration+set+need+to+be) is often cited for the point that a 50-example set gives an agreement estimate with a wide interval, so a single kappa from 50 items can mislead you. We do not rely on that figure here. Instead, treat the size as a budget: decide how narrow an interval you need, and size the set to match. Research is also making smaller sets go further: [Unell et al. derived a Chernoff bound](https://openreview.net/pdf?id=2aBTwSOL0e) that cuts the sample size needed for reliable ICC estimation by an average of 18.3% against a baseline.

What moves the number:

- **Judge quality.** Weaker judges produce noisier agreement, so you need more labels to pin it down.
- **Class balance.** If failures are rare, you need enough of them in the set to estimate the false-negative rate at all. Oversample failures on purpose.
- **Ambiguity.** The more subjective the criterion, the more labels it takes for a stable estimate.
- **Precision you need.** Gating a release on a one-point difference demands a bigger set than spotting a ten-point regression.

## Step 2: Measure agreement the right way

Start with raw percent agreement because it is easy to read, then add a chance-corrected statistic, because raw agreement flatters you when one label dominates. If 95% of outputs are good, a judge that always says "good" scores 95% agreement and tells you nothing.

Chrome's guidance covers the standard choices: [Cohen's kappa for two raters and Fleiss' kappa for three or more](https://developer.chrome.com/docs/ai/evals/expert-judge). Kappa runs from 0 (chance-level agreement) to 1.0 (perfect agreement), and the guidance suggests iterating on the rubric when it falls below about 0.61. Alongside kappa, compute accuracy, precision, recall and F1 for the judge against your human labels.

A minimal scoring pass looks like this:

```python
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

human = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
judge = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

kappa = cohen_kappa_score(human, judge)
prec, rec, f1, _ = precision_recall_fscore_support(human, judge, average="binary")
print(f"kappa={kappa:.2f} precision={prec:.2f} recall={rec:.2f} f1={f1:.2f}")
```

Run it on your holdout, never the tuning data. Then read the confusion matrix, not just the summary. Whether the judge errs toward false passes or false failures changes what you do next.

### When the task is ambiguous

Forced-choice agreement can mislead when reasonable people disagree. A [2025 study on rating indeterminacy](https://arxiv.org/html/2503.05965v4) recommends adding an explicit "Maybe" or "Tie" option where the task allows it, or comparing the continuous distributions of human and judge responses with a metric such as mean squared error. If your labelers split on a case, record the split rather than forcing a single winner.

### Expect uneven results across judges

Agreement between judges is often much lower than teams assume. One 2026 study of human-AI co-creation [reported a mean pairwise Cohen's kappa of 0.1592](https://arxiv.org/html/2604.27727v1) among the judges it evaluated. That is one task and one set of models, not a rule, but it is a reason not to treat any single judge's verdict as ground truth.

## Step 3: Detect and reduce judge bias

Agreement on average can hide structured errors. Test for the known ones directly.

| Bias                          | What it looks like                                                   | How to check                                                          |
| ----------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Position                      | The judge prefers the first (or second) answer regardless of content | Run every pair twice in AB and BA order and count flips               |
| Verbosity                     | Longer answers win                                                   | Compare scores on paired short and long answers with the same content |
| Fluency / superficial quality | Polished prose beats correct but plain prose                         | Add correct-but-rough and wrong-but-polished cases to the set         |
| Leniency                      | The judge passes too much                                            | Check false-positive rate against humans                              |

An [ACL study of evaluation bias](https://aclanthology.org/2024.ccl-1.101/) identifies superficial-quality, verbosity and fluency biases, and uses probability-level and prompt-level calibration to reduce superficial-quality bias.

One finding deserves attention. A 2026 Berkeley study describes a [consistency-bias paradox](https://www.alphaxiv.org/abs/2606.19544): models can show test-retest reliability above 0.95 while carrying position bias above 0.10. In other words, a judge that gives the same answer every time can be consistently wrong in the same direction. Stability is necessary and nowhere near sufficient.

Chrome's guidance, as summarized in the research, points to a minimum protocol that is cheap to adopt: chance-corrected agreement, paired position testing in both orders, and re-judging to check stability.

## Step 4: Calibrate the judge or correct its scores

There are several routes. Pick by what you can change and what you need to report.

**Improve the rubric first.** Most disagreement traces back to vague criteria. Read the cases where judge and humans split, write down why, and tighten the definitions. Add two or three labeled examples that sit on the boundary. Re-measure on the holdout.

**Align automatically.** Some tools learn from human feedback on the same traces. In MLflow, you define a judge with `make_judge`, collect human feedback under the same assessment name as the judge, and run an alignment optimizer so the judge's instructions adapt to your labels. In the alignment workflow we document, the human feedback must share the judge's name, and you need at least 10 traces carrying both.

```python
from typing import Literal
from mlflow.genai.judges import make_judge

quality_judge = make_judge(
    name="product_quality",
    instructions=(
        "Evaluate if the product description in {{ outputs }} "
        "is accurate and helpful for the query in {{ inputs }}. "
        "Rate as: excellent, good, fair, or poor"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor"],
    model="anthropic:/claude-opus-4-1-20250805",
)
```

Ten traces is a floor for the tooling to run, not a calibration set you should trust. Use a larger, separate holdout to measure whether alignment helped.

**Correct the metric statistically.** If you keep the judge fixed, estimate its sensitivity and specificity on the calibration set and adjust the score. [Lee et al. (2026)](https://proceedings.mlr.press/v306/lee26bl.html) propose a plug-in framework that corrects the bias in naive scores using a human-labeled calibration set. Their related work also covers [adaptive allocation of human calibration samples](https://arxiv.org/html/2511.21140v3), so you spend labeling effort where it narrows the interval most.

**Model the judge as an imperfect classifier.** [Feng et al. (2026)](https://arxiv.org/html/2601.20913v1) model the judge's true and false positive rates explicitly and derive variance-corrected thresholds for valid hypothesis tests, even when the judge is noisy. This matters when a test result feeds a safety or compliance decision.

**Calibrate the judge's confidence.** If you want per-verdict uncertainty, [Radharapu et al. (2026)](https://aclanthology.org/2026.acl-industry.14.pdf) trained linear probes on hidden states with Brier score loss, reporting roughly 10x computational savings over multi-generation methods. This needs access to model internals, so it fits self-hosted judges rather than API-only ones.

## Step 5: Report calibrated results honestly

A calibrated score is still an estimate. Report it as one.

- **Show error bars on every metric.** The [2024 paper on adding error bars to evals](https://arxiv.org/html/2411.00640) recommends reporting the number of evaluation questions and the standard error next to each mean, with a normal-approximation 95% interval of the estimate plus or minus 1.96 times its standard error.
- **Carry the judge's uncertainty too.** Lee et al. describe confidence intervals that account for uncertainty in both the test dataset and the calibration set. Their [paper page summary](https://huggingface.co/papers/2511.21140) explains that combining these variance sources with the delta method gives a wider, more honest interval that reflects imperfect sensitivity and specificity.
- **State the calibration conditions.** Name the judge model, prompt version, calibration set size and agreement figures next to the score. A score without them cannot be reproduced.
- **Review a targeted sample by hand.** Princeton's [2026 trends report on agent evaluation](https://sage.cs.princeton.edu/trends/emerging-trends-paper.pdf) recommends validating autograders against representative expert-coded holdouts, manually reviewing a subset of passed and stopped cases, quantifying human-LLM agreement and judge biases, and reflecting judge error in reported uncertainty.

A wider interval can feel like a step backward. It is the opposite: it tells your team whether a two-point gain is real before you ship on it.

## Step 6: Keep calibration current

Calibration decays. Re-run it when any of these change:

1. **The judge model.** A provider update or a swap to a cheaper model changes error rates.
2. **The judge prompt or rubric.** Even small wording edits can move agreement.
3. **The task.** New user intents, languages or tools put outputs outside what your set covered.
4. **The application under test.** A better agent produces harder cases, and a judge calibrated on easy failures will miss subtle ones.

A workable routine: version the judge alongside the prompts, keep a frozen holdout for regression checks, add newly disputed production cases to the labeled set each cycle, and track kappa over time as a metric of its own. When it drops, treat it like any other regression.

## Our perspective: judges are software you have to test

We build MLflow, and the pattern we see most often is teams treating the judge as a fixed measuring instrument. It is a model with a prompt, and it needs the same discipline as the application it grades: versioning, a test set, and a record of what changed.

Three convictions shape how we build.

**Traces are the evidence.** Calibration needs humans and judges to look at the same thing. Deep tracing of agentic reasoning gives both the full trajectory, so a disagreement can be diagnosed rather than argued.

**Feedback should live where the judge lives.** Human labels attached to the same traces and under the same names as judge assessments make alignment a workflow instead of a spreadsheet exercise.

![Our perspective: judges are software you have to test](https://media.babylovegrowth.ai/blog-images/organization-30814/1793256751442_Our-perspective-judges-are-software-you-have-to-test.jpeg)

**No paywalls on trust.** Our positioning is that all features are open source under Linux Foundation governance, with no enterprise feature paywalls. Evaluation you cannot inspect is evaluation you cannot calibrate, so we think the judge, the data and the tooling should be open to inspection.

If you take one habit from this guide, make it this: never publish a judge-based score without the agreement number that earns it.

> _— Kevin_

## Calibrate judges on your own traces with MLflow

MLflow is an open-source platform for GenAI and LLM lifecycle management. It combines tracing of agent behavior, automated evaluation with LLM-as-a-Judge, prompt management and versioning, and an AI Gateway for cross-provider governance, so your calibration loop sits next to the traces it depends on.

![Mlflow](https://csuxjmfbwmkxiegfpljm.supabase.co/storage/v1/object/public/blog-images/organization-30814/1778726621079_mlflow.jpg)

You can start with the free distribution and no account. [Explore MLflow](https://mlflow.org) to set up tracing and build your first judge.

## FAQ

### What kappa score is good enough for an LLM judge?

There is no universal cutoff. Chrome's guidance suggests iterating on the rubric when kappa falls below about 0.61, with 0 meaning chance-level agreement and 1.0 perfect agreement. Also compare against human-human kappa, since a judge cannot be more aligned than the people who labeled the data.

### How many labeled examples do I need?

Enough to give an interval narrow enough for the decision you are making. Small sets produce wide intervals around kappa and accuracy, weaker judges need more labels, and rare failure classes need deliberate oversampling. Report the set size beside every calibrated score.

### Is a consistent judge a reliable judge?

No. Berkeley's 2026 study found models with test-retest reliability above 0.95 that still had position bias above 0.10. Check consistency and validity separately, including AB/BA order tests.

### Do I need to recalibrate after changing the judge model?

Yes. A new model, a new prompt or a new task can all shift sensitivity and specificity. Re-measure agreement on a frozen holdout and update the calibration set with newly disputed cases.

### Can I fix a biased judge without retraining it?

Often, yes. You can tighten the rubric, add boundary examples, test both answer orders, or correct the reported score statistically using sensitivity and specificity estimated from a human-labeled set. Probability-level calibration also helps with some biases.

## Sources

- [How to Correctly Report LLM-as-a-Judge Evaluations (PMLR)](https://proceedings.mlr.press/v306/lee26bl.html)
- [How to Correctly Report LLM-as-a-Judge Evaluations (arXiv)](https://arxiv.org/html/2511.21140v3)
- [Paper page: How to Correctly Report LLM-as-a-Judge Evaluations](https://huggingface.co/papers/2511.21140)
- [Build an expert judge, Chrome for Developers](https://developer.chrome.com/docs/ai/evals/expert-judge)
- [Smarter Sampling for LLM Judges](https://openreview.net/pdf?id=2aBTwSOL0e)
- [Validating LLM-as-a-Judge Systems under Rating Indeterminacy](https://arxiv.org/html/2503.05965v4)
- [LLM-as-a-Judge for Human-AI Co-Creation](https://arxiv.org/html/2604.27727v1)
- [Mitigating the Bias of Large Language Model Evaluation](https://aclanthology.org/2024.ccl-1.101/)
- [Reliability without Validity](https://www.alphaxiv.org/abs/2606.19544)
- [Noisy but Valid: Robust Statistical Evaluation of LLMs](https://arxiv.org/html/2601.20913v1)
- [Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation](https://aclanthology.org/2026.acl-industry.14.pdf)
- [Adding Error Bars to Evals](https://arxiv.org/html/2411.00640)
- [Emerging Trends in AI Agent Evaluation, Princeton](https://sage.cs.princeton.edu/trends/emerging-trends-paper.pdf)
