# Guidelines-based LLM Scorers

[Guidelines](/mlflow-website/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.Guidelines) is a powerful scorer class designed to let you quickly and easily customize evaluation by defining natural language criteria that are framed as pass/fail conditions. It is ideal for checking compliance with rules, style guides, or information inclusion/exclusion.

Guidelines have the distinct advantage of being easy to explain to business stakeholders ("we are evaluating if the app delivers upon this set of rules") and, as such, can often be directly written by domain experts.

### Example usage[​](#example-usage "Direct link to Example usage")

First, define the guidelines as a simple string:

python

```
tone = "The response must maintain a courteous, respectful tone throughout.  It must show empathy for customer concerns."
easy_to_understand = "The response must use clear, concise language and structure responses logically. It must avoid jargon or explain technical terms when used."
banned_topics = "If the request is a question about product pricing, the response must politely decline to answer and refer the user to the pricing page."
```

Then pass each guideline to the `Guidelines` class to create a scorer and run evaluation:

python

```
import mlflow

eval_dataset = [
    {
        "inputs": {"question": "I'm having trouble with my account.  I can't log in."},
        "outputs": "I'm sorry to hear that you're having trouble logging in. Please provide me with your username and the specific issue you're experiencing, and I'll be happy to help you resolve it.",
    },
    {
        "inputs": {"question": "How much does a microwave cost?"},
        "outputs": "The microwave costs $100.",
    },
    {
        "inputs": {"question": "How does a refrigerator work?"},
        "outputs": "A refrigerator operates via thermodynamic vapor-compression cycles utilizing refrigerant phase transitions. The compressor pressurizes vapor which condenses externally, then expands through evaporator coils to absorb internal heat through endothermic vaporization.",
    },
]

mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        # Create a scorer for each guideline
        Guidelines(name="tone", guidelines=tone),
        Guidelines(name="easy_to_understand", guidelines=easy_to_understand),
        Guidelines(name="banned_topics", guidelines=banned_topics),
    ],
)
```

![Guidelines scorers result](/mlflow-website/docs/latest/images/mlflow-3/eval-monitor/scorers/guideline-scorers-results.png)

## Selecting Judge Models[​](#selecting-judge-models "Direct link to Selecting Judge Models")

MLflow supports all major LLM providers, such as OpenAI, Anthropic, Google, xAI, and more.

See [Supported Models](/mlflow-website/docs/latest/genai/eval-monitor/scorers/llm-judge.md#supported-models) for more details.

## Next Steps[​](#next-steps "Direct link to Next Steps")

### [Evaluate Agents](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn how to evaluate AI agents with specialized techniques and scorers](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/agents.md)

### [Evaluate Traces](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Evaluate production traces to understand and improve your AI application's behavior](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

[Learn more →](/mlflow-website/docs/latest/genai/eval-monitor/running-evaluation/traces.md)

### [Collect User Feedback](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Integrate user feedback to continuously improve your evaluation criteria and model performance](/mlflow-website/docs/latest/genai/assessments/feedback.md)

[Learn more →](/mlflow-website/docs/latest/genai/assessments/feedback.md)
