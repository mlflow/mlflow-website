# MLflow Sentence Transformers Flavor

The MLflow Sentence Transformers flavor provides integration with the [Sentence Transformers](https://www.sbert.net/) library for generating semantic embeddings from text.

## Key Features[​](#key-features "Direct link to Key Features")

#### Model Logging

Save and version sentence transformer models with full metadata

#### Embedding Generation

Deploy models as embeddings services with standardized interfaces

#### Semantic Task Support

Handle semantic search, similarity, classification, and clustering tasks

#### PyFunc Integration

Serve models with MLflow's generic Python function interface

## Installation[​](#installation "Direct link to Installation")

bash

```bash
pip install mlflow[sentence-transformers]

```

## Basic Usage[​](#basic-usage "Direct link to Basic Usage")

### Logging and Loading Models[​](#logging-and-loading-models "Direct link to Logging and Loading Models")

python

```python
import mlflow
from sentence_transformers import SentenceTransformer

# Load and log a model
model = SentenceTransformer("all-MiniLM-L6-v2")

with mlflow.start_run():
    model_info = mlflow.sentence_transformers.log_model(
        model=model,
        name="model",
        input_example=["Sample text for inference"],
    )

# Load as native sentence transformer
loaded_model = mlflow.sentence_transformers.load_model(model_info.model_uri)
embeddings = loaded_model.encode(["Hello world", "MLflow is great"])

# Load as PyFunc
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
result = pyfunc_model.predict(["Hello world", "MLflow is great"])

```

### Model Signatures[​](#model-signatures "Direct link to Model Signatures")

Define explicit signatures for production deployments:

python

```python
from mlflow.models import infer_signature

sample_texts = [
    "MLflow makes ML development easier",
    "Sentence transformers create embeddings",
]
sample_embeddings = model.encode(sample_texts)

signature = infer_signature(sample_texts, sample_embeddings)

with mlflow.start_run():
    mlflow.sentence_transformers.log_model(
        model=model,
        name="model",
        signature=signature,
        input_example=sample_texts,
    )

```

## Semantic Search[​](#semantic-search "Direct link to Semantic Search")

Build semantic search systems with tracking:

python

```python
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer, util

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "MLflow helps manage the machine learning lifecycle",
]

with mlflow.start_run():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Log model parameters
    mlflow.log_params(
        {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "corpus_size": len(documents),
        }
    )

    # Encode corpus
    corpus_embeddings = model.encode(documents, convert_to_tensor=True)

    # Save corpus
    corpus_df = pd.DataFrame({"documents": documents})
    corpus_df.to_csv("corpus.csv", index=False)
    mlflow.log_artifact("corpus.csv")

    # Semantic search
    query = "What tools help with ML development?"
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

    # Log model
    mlflow.sentence_transformers.log_model(
        model=model,
        name="search_model",
        input_example=[query],
    )

```

## Fine-tuning[​](#fine-tuning "Direct link to Fine-tuning")

Track fine-tuning experiments:

python

```python
import mlflow
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

train_examples = [
    InputExample(texts=["Python programming", "Coding in Python"], label=0.9),
    InputExample(texts=["Machine learning model", "ML algorithm"], label=0.8),
    InputExample(texts=["Software development", "Cooking recipes"], label=0.1),
]

with mlflow.start_run():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Log training parameters
    mlflow.log_params(
        {
            "base_model": "all-MiniLM-L6-v2",
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
        }
    )

    # Fine-tune
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
    )

    # Log fine-tuned model
    mlflow.sentence_transformers.log_model(
        model=model,
        name="fine_tuned_model",
    )

```

## Tutorials[​](#tutorials "Direct link to Tutorials")

### Quickstart[​](#quickstart "Direct link to Quickstart")

[Sentence Transformers Quickstart](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart.md)

[Learn the basics of using Sentence Transformers with MLflow](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart.md)

### Advanced Tutorials[​](#advanced-tutorials "Direct link to Advanced Tutorials")

[Semantic Similarity](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers.md)

[Determine similarity scores between sentences using embeddings](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-similarity/semantic-similarity-sentence-transformers.md)[](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.md)

[Semantic Search](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.md)

[](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.md)

[Find the most similar embeddings within a corpus of text](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/semantic-search/semantic-search-sentence-transformers.md)[](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.md)

[Paraphrase Mining](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.md)

[](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.md)

[Identify semantically similar sentences in a text corpus](/mlflow-website/docs/latest/ml/deep-learning/sentence-transformers/tutorials/paraphrase-mining/paraphrase-mining-sentence-transformers.md)

## Learn More[​](#learn-more "Direct link to Learn More")

### [Model Registry](/mlflow-website/docs/latest/ml/model-registry.md)

[Version and manage Sentence Transformer models](/mlflow-website/docs/latest/ml/model-registry.md)

[Learn more →](/mlflow-website/docs/latest/ml/model-registry.md)

### [MLflow Tracking](/mlflow-website/docs/latest/ml/tracking.md)

[Track experiments, parameters, and metrics](/mlflow-website/docs/latest/ml/tracking.md)

[Learn more →](/mlflow-website/docs/latest/ml/tracking.md)

### [Sentence Transformers Library](https://www.sbert.net/)

[Official documentation for the Sentence Transformers library](https://www.sbert.net/)

[Learn more →](https://www.sbert.net/)
