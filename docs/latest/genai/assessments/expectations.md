# Ground Truth Expectations

MLflow Expectations provide a systematic way to capture ground truth - the correct or desired outputs that your AI should produce. By establishing these reference points, you create the foundation for meaningful evaluation and continuous improvement of your GenAI applications.

For complete API documentation and implementation details, see the [`mlflow.log_expectation()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_expectation) reference.

## What are Expectations?[​](#what-are-expectations "Direct link to What are Expectations?")

[Expectations](/mlflow-website/docs/latest/genai/concepts/expectations.md) define the "gold standard" for what your AI should produce given specific inputs. They represent the correct answer, desired behavior, or ideal output as determined by domain experts. Think of expectations as the answer key against which actual AI performance is measured.

Unlike [feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md) that evaluates what happened, expectations establish what should happen. They're always created by humans who have the expertise to define correct outcomes.

## Prerequisites[​](#prerequisites "Direct link to Prerequisites")

Before using the [Expectations API](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_expectation), ensure you have:

* MLflow 3.2.0 or later installed
* An active MLflow tracking server or local tracking setup
* Traces that have been logged from your GenAI application to an MLflow Experiment

## Why Annotate Ground Truth?[​](#why-annotate-ground-truth "Direct link to Why Annotate Ground Truth?")

#### Create Evaluation Baselines

Establish reference points for objective accuracy measurement. Without ground truth, you can't measure how well your AI performs against known correct answers.

#### Enable Systematic Testing

Transform ad-hoc testing into systematic evaluation by building datasets of expected outputs to consistently measure performance across versions and configurations.

#### Support Fine-Tuning and Training

Create high-quality training data from ground truth annotations. Essential for fine-tuning models and training automated evaluators.

#### Establish Quality Standards

Codify quality requirements and transform implicit knowledge into explicit, measurable criteria that everyone can understand and follow.

## Types of Expectations[​](#types-of-expectations "Direct link to Types of Expectations")

* Factual
* Structured
* Behavioral
* Span-Level

### Factual Expectations[​](#factual-expectations "Direct link to Factual Expectations")

For questions with definitive answers:

python

```
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_answer",
    value="The speed of light in vacuum is 299,792,458 meters per second",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="physics_expert@university.edu",
    ),
)
```

### Structured Expectations[​](#structured-expectations "Direct link to Structured Expectations")

For complex outputs with multiple components:

python

```
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_extraction",
    value={
        "company": "TechCorp Inc.",
        "sentiment": "positive",
        "key_topics": ["product_launch", "quarterly_earnings", "market_expansion"],
        "action_required": True,
    },
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="business_analyst@company.com"
    ),
)
```

### Behavioral Expectations[​](#behavioral-expectations "Direct link to Behavioral Expectations")

For defining how the AI should act:

python

```
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_behavior",
    value={
        "should_escalate": True,
        "required_elements": ["empathy", "solution_offer", "follow_up"],
        "max_response_length": 150,
        "tone": "professional_friendly",
    },
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="customer_success_lead@company.com",
    ),
)
```

### Span-Level Expectations[​](#span-level-expectations "Direct link to Span-Level Expectations")

For specific operations within your AI pipeline:

python

```
# Expected documents for RAG retrieval
mlflow.log_expectation(
    trace_id=trace_id,
    span_id=retrieval_span_id,
    name="expected_documents",
    value=["policy_doc_2024", "faq_section_3", "user_guide_ch5"],
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="information_architect@company.com",
    ),
)
```

## Step-by-Step Guides[​](#step-by-step-guides "Direct link to Step-by-Step Guides")

### Add Ground Truth Annotation via UI[​](#add-ground-truth-annotation-via-ui "Direct link to Add Ground Truth Annotation via UI")

The MLflow UI provides an intuitive way to add expectations directly to traces. This approach is ideal for domain experts who need to define ground truth without writing code, and for collaborative annotation workflows where multiple stakeholders contribute different perspectives.

Show Step-by-Step Instructions (8 steps)

The expectation will be immediately attached to the trace, establishing the ground truth reference for future evaluation.

### Log Ground Truth via API[​](#log-ground-truth-via-api "Direct link to Log Ground Truth via API")

Use the programmatic [`mlflow.log_expectation()`](/mlflow-website/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_expectation) API when you need to automate expectation creation, integrate with existing annotation tools, or build custom ground truth collection workflows.

* Single Annotations
* Batch Annotations

Programmatically create expectations for systematic ground truth collection:

**1. Set up your annotation environment:**

python

```
import mlflow
from mlflow.entities import AssessmentSource
from mlflow.entities.assessment_source import AssessmentSourceType

# Define your domain expert source
expert_source = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN, source_id="domain_expert@company.com"
)
```

**2. Create expectations for different data types:**

python

```
def log_factual_expectation(trace_id, question, correct_answer):
    """Log expectation for factual questions."""
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_factual_answer",
        value=correct_answer,
        source=expert_source,
        metadata={
            "question": question,
            "expectation_type": "factual",
            "confidence": "high",
            "verified_by": "subject_matter_expert",
        },
    )


def log_structured_expectation(trace_id, expected_extraction):
    """Log expectation for structured data extraction."""
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_extraction",
        value=expected_extraction,
        source=expert_source,
        metadata={
            "expectation_type": "structured",
            "schema_version": "v1.0",
            "annotation_guidelines": "company_extraction_standards_v2",
        },
    )


def log_behavioral_expectation(trace_id, expected_behavior):
    """Log expectation for AI behavior patterns."""
    mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_behavior",
        value=expected_behavior,
        source=expert_source,
        metadata={
            "expectation_type": "behavioral",
            "behavior_category": "customer_service",
            "compliance_requirement": "company_policy_v3",
        },
    )
```

**3. Use the functions in your annotation workflow:**

python

```
# Example: Annotating a customer service interaction
trace_id = "tr-customer-service-001"

# Define what the AI should have said
factual_answer = "Your account balance is $1,234.56 as of today."
log_factual_expectation(trace_id, "What is my account balance?", factual_answer)

# Define expected data extraction
expected_extraction = {
    "intent": "account_balance_inquiry",
    "account_type": "checking",
    "urgency": "low",
    "requires_authentication": True,
}
log_structured_expectation(trace_id, expected_extraction)

# Define expected behavior
expected_behavior = {
    "should_verify_identity": True,
    "tone": "professional_helpful",
    "should_offer_additional_help": True,
    "escalation_required": False,
}
log_behavioral_expectation(trace_id, expected_behavior)
```

For large-scale ground truth collection, use batch annotation:

**1. Define the batch annotation function:**

python

```
def annotate_batch_expectations(annotation_data):
    """Annotate multiple traces with ground truth expectations."""
    for item in annotation_data:
        try:
            mlflow.log_expectation(
                trace_id=item["trace_id"],
                name=item["expectation_name"],
                value=item["expected_value"],
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=item["annotator_id"],
                ),
                metadata={
                    "batch_id": item["batch_id"],
                    "annotation_session": item["session_id"],
                    "quality_checked": True,
                },
            )
            print(f"✓ Annotated {item['trace_id']}")
        except Exception as e:
            print(f"✗ Failed to annotate {item['trace_id']}: {e}")
```

**2. Prepare your annotation data:**

python

```
# Example batch annotation data
batch_data = [
    {
        "trace_id": "tr-001",
        "expectation_name": "expected_answer",
        "expected_value": "Paris is the capital of France",
        "annotator_id": "expert1@company.com",
        "batch_id": "geography_qa_batch_1",
        "session_id": "session_2024_01_15",
    },
    {
        "trace_id": "tr-002",
        "expectation_name": "expected_answer",
        "expected_value": "The speed of light is 299,792,458 m/s",
        "annotator_id": "expert2@company.com",
        "batch_id": "physics_qa_batch_1",
        "session_id": "session_2024_01_15",
    },
]
```

**3. Execute batch annotation:**

python

```
annotate_batch_expectations(batch_data)
```

## Expectation Annotation Workflows[​](#expectation-annotation-workflows "Direct link to Expectation Annotation Workflows")

Different stages of your AI development lifecycle require different approaches to expectation annotation. The following workflows help you systematically create and maintain ground truth expectations that align with your development process and quality goals.

#### Development Phase

Define success criteria by identifying test scenarios, creating expectations with domain experts, testing AI outputs, and iterating on configurations until expectations are met.

#### Production Monitoring

Enable systematic quality tracking by sampling production traces, adding expectations to create evaluation datasets, and tracking performance trends over time.

#### Collaborative Annotation

Use team-based annotation where domain experts define initial expectations, review committees validate and refine, and consensus building resolves disagreements.

## Best Practices[​](#best-practices "Direct link to Best Practices")

### Be Specific and Measurable[​](#be-specific-and-measurable "Direct link to Be Specific and Measurable")

Vague expectations lead to inconsistent evaluation. Define clear, specific criteria that can be objectively verified.

### Document Your Reasoning[​](#document-your-reasoning "Direct link to Document Your Reasoning")

Use metadata to explain why an expectation is defined a certain way:

python

```
mlflow.log_expectation(
    trace_id=trace_id,
    name="expected_diagnosis",
    value={
        "primary": "Type 2 Diabetes",
        "risk_factors": ["obesity", "family_history"],
        "recommended_tests": ["HbA1c", "fasting_glucose"],
    },
    metadata={
        "guideline_version": "ADA_2024",
        "confidence": "high",
        "based_on": "clinical_presentation_and_history",
    },
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="endocrinologist@hospital.org"
    ),
)
```

### Maintain Consistency[​](#maintain-consistency "Direct link to Maintain Consistency")

Use standardized naming and structure across your expectations to enable meaningful analysis and comparison.

## Managing Expectations[​](#managing-expectations "Direct link to Managing Expectations")

Once you've defined expectations for your traces, you may need to retrieve, update, or delete them to maintain accurate ground truth data.

### Retrieving Expectations[​](#retrieving-expectations "Direct link to Retrieving Expectations")

Retrieve specific expectations to analyze your ground truth data:

python

```
# Get a specific expectation by ID
expectation = mlflow.get_assessment(
    trace_id="tr-1234567890abcdef", assessment_id="a-0987654321abcdef"
)

# Access expectation details
name = expectation.name
value = expectation.value
source_type = expectation.source.source_type
metadata = expectation.metadata if hasattr(expectation, "metadata") else None
```

### Updating Expectations[​](#updating-expectations "Direct link to Updating Expectations")

Update existing expectations when ground truth needs refinement:

python

```
from mlflow.entities import Expectation

# Update expectation with corrected information
updated_expectation = Expectation(
    name="expected_answer",
    value="The capital of France is Paris, located in the Île-de-France region",
)

mlflow.update_assessment(
    trace_id="tr-1234567890abcdef",
    assessment_id="a-0987654321abcdef",
    assessment=updated_expectation,
)
```

### Deleting Expectations[​](#deleting-expectations "Direct link to Deleting Expectations")

Remove expectations that were logged incorrectly:

python

```
# Delete specific expectation
mlflow.delete_assessment(
    trace_id="tr-1234567890abcdef", assessment_id="a-5555666677778888"
)
```

## Integration with Evaluation[​](#integration-with-evaluation "Direct link to Integration with Evaluation")

Expectations are most powerful when combined with systematic evaluation:

1. **Automated scoring** against expectations
2. **Human feedback** on expectation achievement
3. **Gap analysis** between expected and actual
4. **Performance metrics** based on expectation matching

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Expectations Concepts](/mlflow-website/docs/latest/genai/concepts/expectations.md)

[Deep dive into expectations architecture and schema](/mlflow-website/docs/latest/genai/concepts/expectations.md)

[Learn more →](/mlflow-website/docs/latest/genai/concepts/expectations.md)

### [Automated and Human Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Learn how to collect quality evaluations from multiple sources](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Start collecting →](/mlflow-website/docs/latest/genai/assessments/feedback.md)

### [LLM Evaluation](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Learn how to systematically evaluate and improve your GenAI applications](/mlflow-website/docs/latest/genai/eval-monitor.md)

[Start evaluating →](/mlflow-website/docs/latest/genai/eval-monitor.md)
