# Feedback Collection

MLflow Feedback provides a comprehensive system for capturing quality evaluations from multiple sources - whether automated AI judges, programmatic rules, or human reviewers. This systematic approach to feedback collection enables you to understand and improve your GenAI application's performance at scale.

For complete API documentation and implementation details, see the [`mlflow.log_feedback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) reference.

## What is Feedback?[​](#what-is-feedback "Direct link to What is Feedback?")

[Feedback](/mlflow-website/docs/latest/genai/concepts/feedback.md) captures evaluations of how well your AI performed. It measures the actual quality of what your AI produced across various dimensions like accuracy, relevance, safety, and helpfulness. Unlike [expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md) that define what should happen, feedback tells you what actually happened and how well it met your quality standards.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Before using feedback collection in MLflow, ensure you have:

* MLflow 3.2.0 or later installed
* An active MLflow tracking server or local tracking setup
* Traces that have been logged from your GenAI application to an MLflow Experiment

## Sources of Feedback[​](#sources-of-feedback "Direct link to Sources of Feedback")

MLflow supports three types of feedback sources, each with unique strengths. You can use a single source or combine multiple sources for comprehensive quality coverage.

#### LLM Judge Evaluation

AI-powered evaluation at scale. LLM judges provide consistent quality assessments for nuanced dimensions like relevance, tone, and safety without human intervention.

#### Programmatic Code Checks

Deterministic rule-based evaluation. Perfect for format validation, compliance checks, and business logic rules that need instant, cost-effective assessment.

#### Human Expert Review

Domain expert evaluation for high-stakes content. Human feedback captures nuanced insights that automated systems miss and serves as the gold standard.

Using the feedback sources within the Python APIs is done as follows:

python

```
from mlflow.entities import AssessmentSource, AssessmentSourceType

# Human expert providing evaluation
human_source = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN, source_id="expert@company.com"
)

# Automated rule-based evaluation
code_source = AssessmentSource(
    source_type=AssessmentSourceType.CODE, source_id="accuracy_checker_v1"
)

# AI-powered evaluation at scale
llm_judge_source = AssessmentSource(
    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4-evaluator"
)
```

## Why Collect Feedback?[​](#why-collect-feedback "Direct link to Why Collect Feedback?")

Collecting feedback on the quality of GenAI applications is critical to a continuous improvement process, ensuring that your application remains effective and is enhanced over time.

#### Enable Continuous Improvement

Create data-driven improvement cycles by systematically collecting quality signals to identify patterns, fix issues, and enhance AI performance over time.

#### Scale Quality Assurance

Monitor quality at production scale by evaluating every trace instead of small samples, catching issues before they impact users.

#### Build Trust Through Transparency

Show stakeholders exactly how quality is measured and by whom, building confidence in your AI system's reliability through clear attribution.

#### Create Training Data

Generate high-quality training datasets from feedback, especially human corrections, to improve both AI applications and evaluation systems.

## How Feedback Works[​](#how-feedback-works "Direct link to How Feedback Works")

### Via API[​](#via-api "Direct link to Via API")

Use the programmatic [`mlflow.log_feedback()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) API when you need to automate feedback collection at scale, integrate with existing systems, or build custom evaluation workflows. The API enables you to collect feedback from all three sources programmatically.

## Step-by-Step Guides[​](#step-by-step-guides "Direct link to Step-by-Step Guides")

### Add Human Evaluation via UI[​](#add-human-evaluation-via-ui "Direct link to Add Human Evaluation via UI")

The MLflow UI provides an intuitive way to add, edit, and manage feedback directly on traces. This approach is ideal for manual review, collaborative evaluation, and situations where domain experts need to provide feedback without writing code.

#### Adding New Feedback[​](#adding-new-feedback "Direct link to Adding New Feedback")

Show Step-by-Step Instructions (8 steps)

The feedback will be immediately attached to the trace with your user information as the source.

#### Editing Existing Feedback[​](#editing-existing-feedback "Direct link to Editing Existing Feedback")

To refine evaluations or correct mistakes:

Show Step-by-Step Instructions (5 steps)

#### Adding Additional Feedback to Existing Entries[​](#adding-additional-feedback-to-existing-entries "Direct link to Adding Additional Feedback to Existing Entries")

When multiple reviewers want to provide feedback on the same aspect, or when you want to add corrections to automated evaluations:

Show Step-by-Step Instructions (4 steps)

This collaborative approach enables multiple perspectives on the same trace aspect, creating richer evaluation datasets and helping identify cases where evaluators disagree.

### Log Automated Assessment via API[​](#log-automated-assessment-via-api "Direct link to Log Automated Assessment via API")

* LLM Judge
* Heuristics Metrics

Implement automated LLM-based evaluation with these steps:

**1. Set up your evaluation environment:**

python

```
import json
import mlflow
from mlflow.entities import AssessmentSource, AssessmentError
from mlflow.entities.assessment_source import AssessmentSourceType
import openai  # or your preferred LLM client

# Configure your LLM client
client = openai.OpenAI(api_key="your-api-key")
```

**2. Create your evaluation prompt:**

python

```
def create_evaluation_prompt(user_input, ai_response):
    return f"""
    Evaluate the AI response for helpfulness and accuracy.

    User Input: {user_input}
    AI Response: {ai_response}

    Rate the response on a scale of 0.0 to 1.0 for:
    1. Helpfulness: How well does it address the user's needs?
    2. Accuracy: Is the information factually correct?

    Respond with only a JSON object:
    {{"helpfulness": 0.0-1.0, "accuracy": 0.0-1.0, "rationale": "explanation"}}
    """
```

**3. Implement the evaluation function:**

python

```
def evaluate_with_llm_judge(trace_id, user_input, ai_response):
    try:
        # Get LLM evaluation
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": create_evaluation_prompt(user_input, ai_response),
                }
            ],
            temperature=0.0,
        )

        # Parse the evaluation

        evaluation = json.loads(response.choices[0].message.content)

        # Log feedback to MLflow
        mlflow.log_feedback(
            trace_id=trace_id,
            name="llm_judge_evaluation",
            value=evaluation,
            rationale=evaluation.get("rationale", ""),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4-evaluator"
            ),
        )

    except Exception as e:
        # Log evaluation failure
        mlflow.log_feedback(
            trace_id=trace_id,
            name="llm_judge_evaluation",
            error=AssessmentError(error_code="EVALUATION_FAILED", error_message=str(e)),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4-evaluator"
            ),
        )
```

**4. Use the evaluation function:**

python

```
# Example usage
trace_id = "your-trace-id"
user_question = "What is the capital of France?"
ai_answer = "The capital of France is Paris."

evaluate_with_llm_judge(trace_id, user_question, ai_answer)
```

Implement programmatic rule-based evaluation:

**1. Define your evaluation rules:**

python

```
def evaluate_response_compliance(response_text):
    """Evaluate response against business rules."""
    results = {
        "has_disclaimer": False,
        "appropriate_length": False,
        "contains_prohibited_terms": False,
        "rationale": [],
    }

    # Check for required disclaimer
    if "This is not financial advice" in response_text:
        results["has_disclaimer"] = True
    else:
        results["rationale"].append("Missing required disclaimer")

    # Check response length
    if 50 <= len(response_text) <= 500:
        results["appropriate_length"] = True
    else:
        results["rationale"].append(
            f"Response length {len(response_text)} outside acceptable range"
        )

    # Check for prohibited terms
    prohibited_terms = ["guaranteed returns", "risk-free", "get rich quick"]
    found_terms = [
        term for term in prohibited_terms if term.lower() in response_text.lower()
    ]
    if found_terms:
        results["contains_prohibited_terms"] = True
        results["rationale"].append(f"Contains prohibited terms: {found_terms}")

    return results
```

**2. Implement the logging function:**

python

```
def log_compliance_check(trace_id, response_text):
    # Run compliance evaluation
    evaluation = evaluate_response_compliance(response_text)

    # Calculate overall compliance score
    compliance_score = (
        sum(
            [
                evaluation["has_disclaimer"],
                evaluation["appropriate_length"],
                not evaluation["contains_prohibited_terms"],
            ]
        )
        / 3
    )

    # Log the feedback
    mlflow.log_feedback(
        trace_id=trace_id,
        name="compliance_check",
        value={"overall_score": compliance_score, "details": evaluation},
        rationale="; ".join(evaluation["rationale"]) or "All compliance checks passed",
        source=AssessmentSource(
            source_type=AssessmentSourceType.CODE, source_id="compliance_validator_v2.1"
        ),
    )
```

**3. Use in your application:**

python

```
# Example usage after your AI generates a response
with mlflow.start_span(name="financial_advice") as span:
    ai_response = your_ai_model.generate(user_question)
    trace_id = span.trace_id

    # Run automated compliance check
    log_compliance_check(trace_id, ai_response)
```

## Managing Feedback[​](#managing-feedback "Direct link to Managing Feedback")

Once you've collected feedback on your traces, you'll need to retrieve, update, and sometimes delete it. These operations are essential for maintaining accurate evaluation data.

### Retrieving Feedback[​](#retrieving-feedback "Direct link to Retrieving Feedback")

Retrieve specific feedback to analyze evaluation results:

python

```
# Get a specific feedback by ID
feedback = mlflow.get_assessment(
    trace_id="tr-1234567890abcdef", assessment_id="a-0987654321abcdef"
)

# Access feedback details
name = feedback.name
value = feedback.value
source_type = feedback.source.source_type
rationale = feedback.rationale if hasattr(feedback, "rationale") else None
```

### Updating Feedback[​](#updating-feedback "Direct link to Updating Feedback")

Update existing feedback when you need to correct or refine evaluations:

python

```
from mlflow.entities import Feedback

# Update feedback with new information
updated_feedback = Feedback(
    name="response_quality",
    value=0.9,
    rationale="Updated after additional review - response is more comprehensive than initially evaluated",
)

mlflow.update_assessment(
    trace_id="tr-1234567890abcdef",
    assessment_id="a-0987654321abcdef",
    assessment=updated_feedback,
)
```

### Deleting Feedback[​](#deleting-feedback "Direct link to Deleting Feedback")

Remove feedback that was logged incorrectly:

python

```
# Delete specific feedback
mlflow.delete_assessment(
    trace_id="tr-1234567890abcdef", assessment_id="a-5555666677778888"
)
```

note

If deleting feedback that has been marked as a replacement using the `override_feedback` API, the original feedback will return to a valid state.

## Overriding Automated Feedback[​](#overriding-automated-feedback "Direct link to Overriding Automated Feedback")

The `override_feedback` function allows human experts to correct automated evaluations while preserving the original for audit trails and learning.

### When to Override vs Update[​](#when-to-override-vs-update "Direct link to When to Override vs Update")

* **Override**: Use when correcting automated feedback - preserves original for analysis
* **Update**: Use when fixing mistakes in existing feedback - modifies in place

### Override Example[​](#override-example "Direct link to Override Example")

python

```
# Step 1: Original automated feedback (logged earlier)
llm_feedback = mlflow.log_feedback(
    trace_id="tr-1234567890abcdef",
    name="relevance",
    value=0.6,
    rationale="Response partially addresses the question",
    source=AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4-evaluator"
    ),
)

# Step 2: Human expert reviews and disagrees
corrected_feedback = mlflow.override_feedback(
    trace_id="tr-1234567890abcdef",
    assessment_id=llm_feedback.assessment_id,
    value=0.9,
    rationale="Response fully addresses the question with comprehensive examples",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="expert_reviewer@company.com"
    ),
    metadata={"override_reason": "LLM underestimated relevance", "confidence": "high"},
)
```

The override process marks the original feedback as invalid but preserves it for historical analysis and model improvement.

## Best Practices[​](#best-practices "Direct link to Best Practices")

### Consistent Naming Conventions[​](#consistent-naming-conventions "Direct link to Consistent Naming Conventions")

Use clear, descriptive names that make feedback data easy to analyze:

python

```
# Good: Descriptive, specific names
mlflow.log_feedback(trace_id=trace_id, name="response_accuracy", value=0.95)
mlflow.log_feedback(trace_id=trace_id, name="sql_syntax_valid", value=True)
mlflow.log_feedback(trace_id=trace_id, name="execution_time_ms", value=245)

# Poor: Vague, inconsistent names
mlflow.log_feedback(trace_id=trace_id, name="good", value=True)
mlflow.log_feedback(trace_id=trace_id, name="score", value=0.95)
```

### Traceable Source Attribution[​](#traceable-source-attribution "Direct link to Traceable Source Attribution")

Provide specific source information for audit trails:

python

```
# Excellent: Version-specific, environment-aware
source = AssessmentSource(
    source_type=AssessmentSourceType.CODE, source_id="response_validator_v2.1_prod"
)

# Good: Individual attribution
source = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN, source_id="expert@company.com"
)

# Poor: Generic, untraceable
source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="validator")
```

### Rich Metadata[​](#rich-metadata "Direct link to Rich Metadata")

Include context that helps with analysis:

python

```
mlflow.log_feedback(
    trace_id=trace_id,
    name="response_quality",
    value=0.85,
    source=human_source,
    metadata={
        "reviewer_expertise": "domain_expert",
        "review_duration_seconds": 45,
        "confidence": "high",
        "criteria_version": "v2.3",
        "evaluation_context": "production_review",
    },
)
```

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Feedback Concepts](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Deep dive into feedback architecture and schema](/mlflow-website/docs/latest/genai/concepts/feedback.md)

[Learn concepts →](/mlflow-website/docs/latest/genai/concepts/feedback.md)

### [Ground Truth Expectations](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Learn how to define expected outputs for evaluation](/mlflow-website/docs/latest/genai/assessments/expectations.md)

[Start annotating →](/mlflow-website/docs/latest/genai/assessments/expectations.md)

### [LLM Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn how to systematically evaluate and improve your GenAI applications](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor.md)
