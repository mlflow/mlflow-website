# Tracing Instructor

![Instructor Tracing via autolog](/mlflow-website/docs/latest/assets/images/instructor-tracing-9d197095322fed09f9b5f0b28577542c.png)

[Instructor](https://python.useinstructor.com/) is an open-source Python library built on top of Pydantic, simplifying structured LLM outputs with validation, retries, and streaming.

[MLflow Tracing](/mlflow-website/docs/latest/genai/tracing.md) works with Instructor by enabling auto-tracing for the underlying LLM libraries. For example, if you use Instructor for OpenAI LLMs, you can enable tracing with `mlflow.openai.autolog()` and the generated traces will capture the structured outputs from Instructor.

Similarly, you can also trace Instructor with other LLM providers, such as Anthropic, Gemini, and LiteLLM, by enabling the corresponding autologging in MLflow.

### Example Usage[​](#example-usage "Direct link to Example Usage")

The following example shows how to trace Instructor call that wraps an OpenAI API.

python

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

# Use other autologging function e.g., mlflow.anthropic.autolog() if you are using Instructor with different LLM providers
mlflow.openai.autolog()

# Optional, create an experiment to store traces
mlflow.set_experiment("Instructor")


# Use Instructor as usual
class ExtractUser(BaseModel):
    name: str
    age: int


client = instructor.from_openai(OpenAI())

res = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
print(f"Name: {res.name}, Age:{res.age}")

```
