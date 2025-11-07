# Fine-Tuning Transformers with MLflow for Enhanced Model Management

[Download this notebook](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/classic-ml/deep-learning/transformers/tutorials/fine-tuning/transformers-fine-tuning.ipynb)

Welcome to our in-depth tutorial on fine-tuning Transformers models with enhanced management using MLflow.

### What You Will Learn in This Tutorial[​](#what-you-will-learn-in-this-tutorial "Direct link to What You Will Learn in This Tutorial")

* Understand the process of fine-tuning a Transformers model.
* Learn to effectively log and manage the training cycle using MLflow.
* Master logging the trained model separately in MLflow.
* Gain insights into using the trained model for practical inference tasks.

Our approach will provide a holistic understanding of model fine-tuning and management, ensuring that you're well-equipped to handle similar tasks in your projects.

#### Emphasizing Fine-Tuning[​](#emphasizing-fine-tuning "Direct link to Emphasizing Fine-Tuning")

Fine-tuning pre-trained models is a common practice in machine learning, especially in the field of NLP. It involves adjusting a pre-trained model to make it more suitable for a specific task. This process is essential as it allows the leveraging of pre-existing knowledge in the model, significantly improving performance on specific datasets or tasks.

#### Role of MLflow in Model Lifecycle[​](#role-of-mlflow-in-model-lifecycle "Direct link to Role of MLflow in Model Lifecycle")

Integrating MLflow in this process is crucial for:

* **Training Cycle Logging**: Keeping a detailed log of the training cycle, including parameters, metrics, and intermediate results.
* **Model Logging and Management**: Separately logging the trained model, tracking its versions, and managing its lifecycle post-training.
* **Inference and Deployment**: Using the logged model for inference, ensuring easy transition from training to deployment.

python

```
# Disable tokenizers warnings when constructing pipelines
%env TOKENIZERS_PARALLELISM=false

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)
```

```
env: TOKENIZERS_PARALLELISM=false
```

### Preparing the Dataset and Environment for Fine-Tuning[​](#preparing-the-dataset-and-environment-for-fine-tuning "Direct link to Preparing the Dataset and Environment for Fine-Tuning")

#### Key Steps in this Section[​](#key-steps-in-this-section "Direct link to Key Steps in this Section")

1. **Loading the Dataset**: Utilizing the `sms_spam` dataset for spam detection.
2. **Splitting the Dataset**: Dividing the dataset into training and test sets with an 80/20 distribution.
3. **Importing Necessary Libraries**: Including libraries like `evaluate`, `mlflow`, `numpy`, and essential components from the `transformers` library.

Before diving into the fine-tuning process, setting up our environment and preparing the dataset is crucial. This step involves loading the dataset, splitting it into training and testing sets, and initializing essential components of the Transformers library. These preparatory steps lay the groundwork for an efficient fine-tuning process.

This setup ensures that we have a solid foundation for fine-tuning our model, with all the necessary data and tools at our disposal. In the following Python code, we'll execute these steps to kickstart our model fine-tuning journey.

python

```
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  pipeline,
)

import mlflow

# Load the "sms_spam" dataset.
sms_dataset = load_dataset("sms_spam")

# Split train/test by an 8/2 ratio.
sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
train_dataset = sms_train_test["train"]
test_dataset = sms_train_test["test"]
```

```
Found cached dataset sms_spam (/Users/benjamin.wilson/.cache/huggingface/datasets/sms_spam/plain_text/1.0.0/53f051d3b5f62d99d61792c91acefe4f1577ad3e4c216fb0ad39e30b9f20019c)
```

```
  0%|          | 0/1 [00:00<?, ?it/s]
```

### Tokenization and Dataset Preparation[​](#tokenization-and-dataset-preparation "Direct link to Tokenization and Dataset Preparation")

In the next code block, we tokenize our text data, preparing it for the fine-tuning process of our model.

With our dataset loaded and split, the next step is to prepare our text data for the model. This involves tokenizing the text, a crucial process in NLP where text is converted into a format that's understandable and usable by our model.

#### Tokenization Process[​](#tokenization-process "Direct link to Tokenization Process")

* **Loading the Tokenizer**: Using the `AutoTokenizer` from the `transformers` library for the `distilbert-base-uncased` model's tokenizer.
* **Defining the Tokenization Function**: Creating a function to tokenize text data, including padding and truncation.
* **Applying Tokenization to the Dataset**: Processing both the training and testing sets for model readiness.

Tokenization is a critical step in preparing text data for NLP tasks. It ensures that the data is in a format that the model can process, and by handling aspects like padding and truncation, it ensures consistency across our dataset, which is vital for training stability and model performance.

python

```
# Load the tokenizer for "distilbert-base-uncased" model.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
  # Pad/truncate each text to 512 tokens. Enforcing the same shape
  # could make the training faster.
  return tokenizer(
      examples["sms"],
      padding="max_length",
      truncation=True,
      max_length=128,
  )


seed = 22

# Tokenize the train and test datasets
train_tokenized = train_dataset.map(tokenize_function)
train_tokenized = train_tokenized.remove_columns(["sms"]).shuffle(seed=seed)

test_tokenized = test_dataset.map(tokenize_function)
test_tokenized = test_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
```

```
Map:   0%|          | 0/4459 [00:00<?, ? examples/s]
```

```
Map:   0%|          | 0/1115 [00:00<?, ? examples/s]
```

### Model Initialization and Label Mapping[​](#model-initialization-and-label-mapping "Direct link to Model Initialization and Label Mapping")

Next, we'll set up label mappings and initialize the model for our text classification task.

Having prepared our data, the next crucial step is to initialize our model and set up label mappings. This involves defining a clear relationship between the labels in our dataset and their corresponding representations in the model.

#### Setting Up Label Mappings[​](#setting-up-label-mappings "Direct link to Setting Up Label Mappings")

* **Defining Label Mappings**: Creating bi-directional mappings between integer labels and textual representations ("ham" and "spam").

#### Initializing the Model[​](#initializing-the-model "Direct link to Initializing the Model")

* **Model Selection**: Choosing the `distilbert-base-uncased` model for its balance of performance and efficiency.
* **Model Configuration**: Configuring the model for sequence classification with the defined label mappings.

Proper model initialization and label mapping are key to ensuring that the model accurately understands and processes the task at hand. By explicitly defining these mappings and selecting an appropriate pre-trained model, we lay the groundwork for effective and efficient fine-tuning.

python

```
# Set the mapping between int label and its meaning.
id2label = {0: "ham", 1: "spam"}
label2id = {"ham": 0, "spam": 1}

# Acquire the model from the Hugging Face Hub, providing label and id mappings so that both we and the model can 'speak' the same language.
model = AutoModelForSequenceClassification.from_pretrained(
  "distilbert-base-uncased",
  num_labels=2,
  label2id=label2id,
  id2label=id2label,
)
```

```
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

### Setting Up Evaluation Metrics[​](#setting-up-evaluation-metrics "Direct link to Setting Up Evaluation Metrics")

Next, we focus on defining and computing evaluation metrics to measure our model's performance accurately.

After initializing our model, the next critical step is to define how we'll evaluate its performance. Accurate evaluation is key to understanding how well our model is learning and performing on the task.

#### Choosing and Loading the Metric[​](#choosing-and-loading-the-metric "Direct link to Choosing and Loading the Metric")

* **Metric Selection**: Opting for 'accuracy' as the evaluation metric.
* **Loading the Metric**: Utilizing the `evaluate` library to load the 'accuracy' metric.

#### Defining the Metric Computation Function[​](#defining-the-metric-computation-function "Direct link to Defining the Metric Computation Function")

* **Function for Metric Computation**: Creating a function, `compute_metrics`, for calculating accuracy during model evaluation.
* **Processing Predictions**: Handling logits and labels from predictions to compute accuracy.

Properly setting up evaluation metrics allows us to objectively measure the model's performance. By using standardized metrics, we can compare our model's performance against benchmarks or other models, ensuring that our fine-tuning process is effective and moving in the right direction.

python

```
# Define the target optimization metric
metric = evaluate.load("accuracy")


# Define a function for calculating our defined target optimization metric during training
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)
```

### Configuring the Training Environment[​](#configuring-the-training-environment "Direct link to Configuring the Training Environment")

In this step, we're going to configure our Trainer, supplying important training configurations via the use of the `TrainingArguments` API.

With our model and metrics ready, the next important step is to configure the training environment. This involves setting up the training arguments and initializing the Trainer, a component that orchestrates the model training process.

#### Training Arguments Configuration[​](#training-arguments-configuration "Direct link to Training Arguments Configuration")

* **Defining the Output Directory**: We specify the `training_output_dir` where our model checkpoints will be saved during training. This helps in managing and storing model states at different stages of training.
* **Specifying Training Arguments**: We create an instance of `TrainingArguments` to define various parameters for training, such as the output directory, evaluation strategy, batch sizes for training and evaluation, logging frequency, and the number of training epochs. These parameters are critical for controlling how the model is trained and evaluated.

#### Initializing the Trainer[​](#initializing-the-trainer "Direct link to Initializing the Trainer")

* **Creating the Trainer Instance**: We use the Trainer class from the Transformers library, providing it with our model, the previously defined training arguments, datasets for training and evaluation, and the function to compute metrics.
* **Role of the Trainer**: The Trainer handles all aspects of training and evaluating the model, including the execution of training loops, handling of data batching, and calling the compute metrics function. It simplifies the training process, making it more streamlined and efficient.

#### Importance of Proper Training Configuration[​](#importance-of-proper-training-configuration "Direct link to Importance of Proper Training Configuration")

Setting up the training environment correctly is essential for effective model training. Proper configuration ensures that the model is trained under optimal conditions, leading to better performance and more reliable results.

In the following code block, we'll configure our training environment and initialize the Trainer, setting the stage for the actual training process.

python

```
# Checkpoints will be output to this `training_output_dir`.
training_output_dir = "/tmp/sms_trainer"
training_args = TrainingArguments(
  output_dir=training_output_dir,
  evaluation_strategy="epoch",
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  logging_steps=8,
  num_train_epochs=3,
)

# Instantiate a `Trainer` instance that will be used to initiate a training run.
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_tokenized,
  eval_dataset=test_tokenized,
  compute_metrics=compute_metrics,
)
```

python

```
# If you are running this tutorial in local mode, leave the next line commented out.
# Otherwise, uncomment the following line and set your tracking uri to your local or remote tracking server.

# mlflow.set_tracking_uri("http://127.0.0.1:8080")
```

### Integrating MLflow for Experiment Tracking[​](#integrating-mlflow-for-experiment-tracking "Direct link to Integrating MLflow for Experiment Tracking")

The final preparatory step before beginning the training process is to integrate MLflow for experiment tracking.

MLflow is a critical tool in our workflow, enabling us to log, monitor, and compare different runs of our model training.

#### Setting up the MLflow Experiment[​](#setting-up-the-mlflow-experiment "Direct link to Setting up the MLflow Experiment")

* **Naming the Experiment**: We use `mlflow.set_experiment` to create a new experiment or assign the current run to an existing experiment. In this case, we name our experiment "Spam Classifier Training". This name should be descriptive and related to the task at hand, aiding in organizing and identifying experiments later.
* **Role of MLflow in Training**: By setting up an MLflow experiment, we can track various aspects of our model training, such as parameters, metrics, and outputs. This tracking is invaluable for comparing different models, tuning hyperparameters, and maintaining a record of our experiments.

#### Benefits of Experiment Tracking[​](#benefits-of-experiment-tracking "Direct link to Benefits of Experiment Tracking")

Utilizing MLflow for experiment tracking offers several advantages:

* **Organization**: Keeps your training runs organized and easily accessible.
* **Comparability**: Allows for easy comparison of different training runs to understand the impact of changes in parameters or data.
* **Reproducibility**: Enhances the reproducibility of experiments by logging all necessary details.

With MLflow set up, we're now ready to begin the training process, keeping track of every important aspect along the way.

In the next code snippet, we'll set up our MLflow experiment for tracking the training of our spam classification model.

python

```
# Pick a name that you like and reflects the nature of the runs that you will be recording to the experiment.
mlflow.set_experiment("Spam Classifier Training")
```

```
<Experiment: artifact_location='file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/transformers/tutorials/fine-tuning/mlruns/258758267044147956', creation_time=1701291176206, experiment_id='258758267044147956', last_update_time=1701291176206, lifecycle_stage='active', name='Spam Classifier Training', tags={}>
```

### Starting the Training Process with MLflow[​](#starting-the-training-process-with-mlflow "Direct link to Starting the Training Process with MLflow")

In this step, we initiate the fine-tuning training run, utilizing the native auto-logging functionality to record the parameters used and loss metrics calculated during the training process.

With our model, training arguments, and MLflow experiment set up, we are now ready to start the actual training process. This step involves initiating an MLflow run, which will encapsulate all the training activities and metrics.

#### Initiating the MLflow Run[​](#initiating-the-mlflow-run "Direct link to Initiating the MLflow Run")

* **Starting an MLflow Run**: We use `mlflow.start_run()` to begin a new MLflow run. This function creates a new run context, under which all the training operations and logging will occur.
* **Training the Model**: Inside the MLflow run context, we call `trainer.train()` to start training our model. This function will run the training loop, processing the data in batches, updating model parameters, and evaluating the model.

#### Monitoring the Training Progress[​](#monitoring-the-training-progress "Direct link to Monitoring the Training Progress")

During training, the `Trainer` object will output logs that provide valuable insights into the training progress:

* **Loss**: Indicates the model's performance, with lower values signifying better performance.
* **Learning Rate**: Shows the current learning rate used during training.
* **Epoch Progress**: Displays the progress through the current epoch.

These logs are crucial for monitoring the model's learning process and making any necessary adjustments. By tracking these metrics within an MLflow run, we can maintain a comprehensive record of the training process, enhancing reproducibility and analysis.

In the next code block, we will start our MLflow run and begin training our model, closely observing the output to gauge the training progress.

python

```
with mlflow.start_run() as run:
  trainer.train()
```

```
  0%|          | 0/1674 [00:00<?, ?it/s]
```

```
{'loss': 0.4891, 'learning_rate': 4.9761051373954604e-05, 'epoch': 0.01}
{'loss': 0.2662, 'learning_rate': 4.95221027479092e-05, 'epoch': 0.03}
{'loss': 0.1756, 'learning_rate': 4.92831541218638e-05, 'epoch': 0.04}
{'loss': 0.107, 'learning_rate': 4.90442054958184e-05, 'epoch': 0.06}
{'loss': 0.0831, 'learning_rate': 4.8805256869773e-05, 'epoch': 0.07}
{'loss': 0.0688, 'learning_rate': 4.8566308243727596e-05, 'epoch': 0.09}
{'loss': 0.0959, 'learning_rate': 4.83273596176822e-05, 'epoch': 0.1}
{'loss': 0.0831, 'learning_rate': 4.80884109916368e-05, 'epoch': 0.11}
{'loss': 0.1653, 'learning_rate': 4.78494623655914e-05, 'epoch': 0.13}
{'loss': 0.1865, 'learning_rate': 4.7610513739546e-05, 'epoch': 0.14}
{'loss': 0.0887, 'learning_rate': 4.73715651135006e-05, 'epoch': 0.16}
{'loss': 0.1009, 'learning_rate': 4.71326164874552e-05, 'epoch': 0.17}
{'loss': 0.1017, 'learning_rate': 4.6893667861409805e-05, 'epoch': 0.19}
{'loss': 0.0057, 'learning_rate': 4.66547192353644e-05, 'epoch': 0.2}
{'loss': 0.0157, 'learning_rate': 4.6415770609319e-05, 'epoch': 0.22}
{'loss': 0.0302, 'learning_rate': 4.61768219832736e-05, 'epoch': 0.23}
{'loss': 0.0013, 'learning_rate': 4.59378733572282e-05, 'epoch': 0.24}
{'loss': 0.0863, 'learning_rate': 4.56989247311828e-05, 'epoch': 0.26}
{'loss': 0.1122, 'learning_rate': 4.54599761051374e-05, 'epoch': 0.27}
{'loss': 0.1092, 'learning_rate': 4.5221027479092e-05, 'epoch': 0.29}
{'loss': 0.0853, 'learning_rate': 4.49820788530466e-05, 'epoch': 0.3}
{'loss': 0.1852, 'learning_rate': 4.4743130227001195e-05, 'epoch': 0.32}
{'loss': 0.0913, 'learning_rate': 4.4504181600955796e-05, 'epoch': 0.33}
{'loss': 0.0232, 'learning_rate': 4.42652329749104e-05, 'epoch': 0.34}
{'loss': 0.0888, 'learning_rate': 4.402628434886499e-05, 'epoch': 0.36}
{'loss': 0.195, 'learning_rate': 4.378733572281959e-05, 'epoch': 0.37}
{'loss': 0.0198, 'learning_rate': 4.3548387096774194e-05, 'epoch': 0.39}
{'loss': 0.056, 'learning_rate': 4.3309438470728796e-05, 'epoch': 0.4}
{'loss': 0.1656, 'learning_rate': 4.307048984468339e-05, 'epoch': 0.42}
{'loss': 0.0032, 'learning_rate': 4.283154121863799e-05, 'epoch': 0.43}
{'loss': 0.1277, 'learning_rate': 4.259259259259259e-05, 'epoch': 0.44}
{'loss': 0.0029, 'learning_rate': 4.2353643966547194e-05, 'epoch': 0.46}
{'loss': 0.1007, 'learning_rate': 4.2114695340501795e-05, 'epoch': 0.47}
{'loss': 0.0038, 'learning_rate': 4.1875746714456396e-05, 'epoch': 0.49}
{'loss': 0.0035, 'learning_rate': 4.1636798088411e-05, 'epoch': 0.5}
{'loss': 0.0015, 'learning_rate': 4.13978494623656e-05, 'epoch': 0.52}
{'loss': 0.1423, 'learning_rate': 4.115890083632019e-05, 'epoch': 0.53}
{'loss': 0.0316, 'learning_rate': 4.0919952210274794e-05, 'epoch': 0.54}
{'loss': 0.0012, 'learning_rate': 4.0681003584229395e-05, 'epoch': 0.56}
{'loss': 0.0009, 'learning_rate': 4.0442054958183996e-05, 'epoch': 0.57}
{'loss': 0.1287, 'learning_rate': 4.020310633213859e-05, 'epoch': 0.59}
{'loss': 0.0893, 'learning_rate': 3.996415770609319e-05, 'epoch': 0.6}
{'loss': 0.0021, 'learning_rate': 3.972520908004779e-05, 'epoch': 0.62}
{'loss': 0.0031, 'learning_rate': 3.9486260454002395e-05, 'epoch': 0.63}
{'loss': 0.0022, 'learning_rate': 3.924731182795699e-05, 'epoch': 0.65}
{'loss': 0.0008, 'learning_rate': 3.900836320191159e-05, 'epoch': 0.66}
{'loss': 0.1119, 'learning_rate': 3.876941457586619e-05, 'epoch': 0.67}
{'loss': 0.0012, 'learning_rate': 3.8530465949820786e-05, 'epoch': 0.69}
{'loss': 0.2618, 'learning_rate': 3.829151732377539e-05, 'epoch': 0.7}
{'loss': 0.0018, 'learning_rate': 3.805256869772999e-05, 'epoch': 0.72}
{'loss': 0.0736, 'learning_rate': 3.781362007168459e-05, 'epoch': 0.73}
{'loss': 0.0126, 'learning_rate': 3.7574671445639184e-05, 'epoch': 0.75}
{'loss': 0.2125, 'learning_rate': 3.7335722819593785e-05, 'epoch': 0.76}
{'loss': 0.0018, 'learning_rate': 3.7096774193548386e-05, 'epoch': 0.77}
{'loss': 0.1386, 'learning_rate': 3.685782556750299e-05, 'epoch': 0.79}
{'loss': 0.0024, 'learning_rate': 3.661887694145759e-05, 'epoch': 0.8}
{'loss': 0.0016, 'learning_rate': 3.637992831541219e-05, 'epoch': 0.82}
{'loss': 0.0011, 'learning_rate': 3.614097968936679e-05, 'epoch': 0.83}
{'loss': 0.0307, 'learning_rate': 3.590203106332139e-05, 'epoch': 0.85}
{'loss': 0.0007, 'learning_rate': 3.566308243727599e-05, 'epoch': 0.86}
{'loss': 0.005, 'learning_rate': 3.542413381123059e-05, 'epoch': 0.87}
{'loss': 0.0534, 'learning_rate': 3.518518518518519e-05, 'epoch': 0.89}
{'loss': 0.0155, 'learning_rate': 3.494623655913979e-05, 'epoch': 0.9}
{'loss': 0.0136, 'learning_rate': 3.4707287933094385e-05, 'epoch': 0.92}
{'loss': 0.1108, 'learning_rate': 3.4468339307048986e-05, 'epoch': 0.93}
{'loss': 0.0017, 'learning_rate': 3.422939068100359e-05, 'epoch': 0.95}
{'loss': 0.0009, 'learning_rate': 3.399044205495819e-05, 'epoch': 0.96}
{'loss': 0.0008, 'learning_rate': 3.375149342891278e-05, 'epoch': 0.97}
{'loss': 0.0846, 'learning_rate': 3.3512544802867384e-05, 'epoch': 0.99}
```

```
  0%|          | 0/140 [00:00<?, ?it/s]
```

```
{'eval_loss': 0.03877367451786995, 'eval_accuracy': 0.9919282511210762, 'eval_runtime': 5.0257, 'eval_samples_per_second': 221.862, 'eval_steps_per_second': 27.857, 'epoch': 1.0}
{'loss': 0.109, 'learning_rate': 3.3273596176821985e-05, 'epoch': 1.0}
{'loss': 0.0084, 'learning_rate': 3.3034647550776586e-05, 'epoch': 1.02}
{'loss': 0.0014, 'learning_rate': 3.279569892473118e-05, 'epoch': 1.03}
{'loss': 0.0008, 'learning_rate': 3.255675029868578e-05, 'epoch': 1.05}
{'loss': 0.0006, 'learning_rate': 3.231780167264038e-05, 'epoch': 1.06}
{'loss': 0.0005, 'learning_rate': 3.207885304659498e-05, 'epoch': 1.08}
{'loss': 0.0004, 'learning_rate': 3.183990442054958e-05, 'epoch': 1.09}
{'loss': 0.0518, 'learning_rate': 3.160095579450418e-05, 'epoch': 1.1}
{'loss': 0.0005, 'learning_rate': 3.136200716845878e-05, 'epoch': 1.12}
{'loss': 0.149, 'learning_rate': 3.112305854241338e-05, 'epoch': 1.13}
{'loss': 0.0022, 'learning_rate': 3.0884109916367984e-05, 'epoch': 1.15}
{'loss': 0.0013, 'learning_rate': 3.0645161290322585e-05, 'epoch': 1.16}
{'loss': 0.0051, 'learning_rate': 3.0406212664277183e-05, 'epoch': 1.18}
{'loss': 0.0005, 'learning_rate': 3.016726403823178e-05, 'epoch': 1.19}
{'loss': 0.0026, 'learning_rate': 2.9928315412186382e-05, 'epoch': 1.2}
{'loss': 0.0005, 'learning_rate': 2.9689366786140983e-05, 'epoch': 1.22}
{'loss': 0.0871, 'learning_rate': 2.9450418160095584e-05, 'epoch': 1.23}
{'loss': 0.0004, 'learning_rate': 2.921146953405018e-05, 'epoch': 1.25}
{'loss': 0.0004, 'learning_rate': 2.897252090800478e-05, 'epoch': 1.26}
{'loss': 0.0003, 'learning_rate': 2.873357228195938e-05, 'epoch': 1.28}
{'loss': 0.0003, 'learning_rate': 2.8494623655913982e-05, 'epoch': 1.29}
{'loss': 0.0003, 'learning_rate': 2.8255675029868577e-05, 'epoch': 1.3}
{'loss': 0.0478, 'learning_rate': 2.8016726403823178e-05, 'epoch': 1.32}
{'loss': 0.0002, 'learning_rate': 2.777777777777778e-05, 'epoch': 1.33}
{'loss': 0.0002, 'learning_rate': 2.753882915173238e-05, 'epoch': 1.35}
{'loss': 0.0003, 'learning_rate': 2.7299880525686978e-05, 'epoch': 1.36}
{'loss': 0.0002, 'learning_rate': 2.706093189964158e-05, 'epoch': 1.38}
{'loss': 0.0005, 'learning_rate': 2.682198327359618e-05, 'epoch': 1.39}
{'loss': 0.0002, 'learning_rate': 2.6583034647550775e-05, 'epoch': 1.41}
{'loss': 0.0003, 'learning_rate': 2.6344086021505376e-05, 'epoch': 1.42}
{'loss': 0.0002, 'learning_rate': 2.6105137395459977e-05, 'epoch': 1.43}
{'loss': 0.0002, 'learning_rate': 2.586618876941458e-05, 'epoch': 1.45}
{'loss': 0.0002, 'learning_rate': 2.5627240143369173e-05, 'epoch': 1.46}
{'loss': 0.0007, 'learning_rate': 2.5388291517323774e-05, 'epoch': 1.48}
{'loss': 0.1336, 'learning_rate': 2.5149342891278375e-05, 'epoch': 1.49}
{'loss': 0.0004, 'learning_rate': 2.4910394265232977e-05, 'epoch': 1.51}
{'loss': 0.0671, 'learning_rate': 2.4671445639187578e-05, 'epoch': 1.52}
{'loss': 0.0004, 'learning_rate': 2.4432497013142176e-05, 'epoch': 1.53}
{'loss': 0.1246, 'learning_rate': 2.4193548387096777e-05, 'epoch': 1.55}
{'loss': 0.1142, 'learning_rate': 2.3954599761051375e-05, 'epoch': 1.56}
{'loss': 0.002, 'learning_rate': 2.3715651135005976e-05, 'epoch': 1.58}
{'loss': 0.002, 'learning_rate': 2.3476702508960574e-05, 'epoch': 1.59}
{'loss': 0.0009, 'learning_rate': 2.3237753882915175e-05, 'epoch': 1.61}
{'loss': 0.0778, 'learning_rate': 2.2998805256869773e-05, 'epoch': 1.62}
{'loss': 0.0007, 'learning_rate': 2.2759856630824374e-05, 'epoch': 1.63}
{'loss': 0.0008, 'learning_rate': 2.2520908004778972e-05, 'epoch': 1.65}
{'loss': 0.0009, 'learning_rate': 2.2281959378733573e-05, 'epoch': 1.66}
{'loss': 0.1032, 'learning_rate': 2.2043010752688174e-05, 'epoch': 1.68}
{'loss': 0.0014, 'learning_rate': 2.1804062126642775e-05, 'epoch': 1.69}
{'loss': 0.001, 'learning_rate': 2.1565113500597373e-05, 'epoch': 1.71}
{'loss': 0.1199, 'learning_rate': 2.132616487455197e-05, 'epoch': 1.72}
{'loss': 0.0009, 'learning_rate': 2.1087216248506572e-05, 'epoch': 1.73}
{'loss': 0.0011, 'learning_rate': 2.084826762246117e-05, 'epoch': 1.75}
{'loss': 0.0007, 'learning_rate': 2.060931899641577e-05, 'epoch': 1.76}
{'loss': 0.0006, 'learning_rate': 2.037037037037037e-05, 'epoch': 1.78}
{'loss': 0.0004, 'learning_rate': 2.013142174432497e-05, 'epoch': 1.79}
{'loss': 0.0005, 'learning_rate': 1.989247311827957e-05, 'epoch': 1.81}
{'loss': 0.1246, 'learning_rate': 1.9653524492234173e-05, 'epoch': 1.82}
{'loss': 0.0974, 'learning_rate': 1.941457586618877e-05, 'epoch': 1.84}
{'loss': 0.0003, 'learning_rate': 1.9175627240143372e-05, 'epoch': 1.85}
{'loss': 0.0007, 'learning_rate': 1.893667861409797e-05, 'epoch': 1.86}
{'loss': 0.1998, 'learning_rate': 1.869772998805257e-05, 'epoch': 1.88}
{'loss': 0.0426, 'learning_rate': 1.845878136200717e-05, 'epoch': 1.89}
{'loss': 0.002, 'learning_rate': 1.821983273596177e-05, 'epoch': 1.91}
{'loss': 0.0009, 'learning_rate': 1.7980884109916368e-05, 'epoch': 1.92}
{'loss': 0.0027, 'learning_rate': 1.774193548387097e-05, 'epoch': 1.94}
{'loss': 0.0004, 'learning_rate': 1.7502986857825567e-05, 'epoch': 1.95}
{'loss': 0.0003, 'learning_rate': 1.7264038231780168e-05, 'epoch': 1.96}
{'loss': 0.1081, 'learning_rate': 1.702508960573477e-05, 'epoch': 1.98}
{'loss': 0.0005, 'learning_rate': 1.678614097968937e-05, 'epoch': 1.99}
```

```
  0%|          | 0/140 [00:00<?, ?it/s]
```

```
{'eval_loss': 0.014878345653414726, 'eval_accuracy': 0.9973094170403587, 'eval_runtime': 4.0209, 'eval_samples_per_second': 277.3, 'eval_steps_per_second': 34.818, 'epoch': 2.0}
{'loss': 0.0005, 'learning_rate': 1.6547192353643968e-05, 'epoch': 2.01}
{'loss': 0.0005, 'learning_rate': 1.630824372759857e-05, 'epoch': 2.02}
{'loss': 0.0004, 'learning_rate': 1.6069295101553167e-05, 'epoch': 2.04}
{'loss': 0.0005, 'learning_rate': 1.5830346475507768e-05, 'epoch': 2.05}
{'loss': 0.0004, 'learning_rate': 1.5591397849462366e-05, 'epoch': 2.06}
{'loss': 0.0135, 'learning_rate': 1.5352449223416964e-05, 'epoch': 2.08}
{'loss': 0.0014, 'learning_rate': 1.5113500597371565e-05, 'epoch': 2.09}
{'loss': 0.0003, 'learning_rate': 1.4874551971326165e-05, 'epoch': 2.11}
{'loss': 0.0003, 'learning_rate': 1.4635603345280766e-05, 'epoch': 2.12}
{'loss': 0.0002, 'learning_rate': 1.4396654719235364e-05, 'epoch': 2.14}
{'loss': 0.0002, 'learning_rate': 1.4157706093189965e-05, 'epoch': 2.15}
{'loss': 0.0003, 'learning_rate': 1.3918757467144564e-05, 'epoch': 2.16}
{'loss': 0.0008, 'learning_rate': 1.3679808841099166e-05, 'epoch': 2.18}
{'loss': 0.0002, 'learning_rate': 1.3440860215053763e-05, 'epoch': 2.19}
{'loss': 0.0002, 'learning_rate': 1.3201911589008365e-05, 'epoch': 2.21}
{'loss': 0.0003, 'learning_rate': 1.2962962962962962e-05, 'epoch': 2.22}
{'loss': 0.0002, 'learning_rate': 1.2724014336917564e-05, 'epoch': 2.24}
{'loss': 0.0002, 'learning_rate': 1.2485065710872163e-05, 'epoch': 2.25}
{'loss': 0.0002, 'learning_rate': 1.2246117084826763e-05, 'epoch': 2.27}
{'loss': 0.0006, 'learning_rate': 1.2007168458781362e-05, 'epoch': 2.28}
{'loss': 0.0875, 'learning_rate': 1.1768219832735962e-05, 'epoch': 2.29}
{'loss': 0.0002, 'learning_rate': 1.1529271206690561e-05, 'epoch': 2.31}
{'loss': 0.0003, 'learning_rate': 1.129032258064516e-05, 'epoch': 2.32}
{'loss': 0.0002, 'learning_rate': 1.1051373954599762e-05, 'epoch': 2.34}
{'loss': 0.0002, 'learning_rate': 1.0812425328554361e-05, 'epoch': 2.35}
{'loss': 0.0003, 'learning_rate': 1.0573476702508961e-05, 'epoch': 2.37}
{'loss': 0.0006, 'learning_rate': 1.033452807646356e-05, 'epoch': 2.38}
{'loss': 0.0002, 'learning_rate': 1.009557945041816e-05, 'epoch': 2.39}
{'loss': 0.0002, 'learning_rate': 9.856630824372761e-06, 'epoch': 2.41}
{'loss': 0.0002, 'learning_rate': 9.61768219832736e-06, 'epoch': 2.42}
{'loss': 0.0002, 'learning_rate': 9.37873357228196e-06, 'epoch': 2.44}
{'loss': 0.0002, 'learning_rate': 9.13978494623656e-06, 'epoch': 2.45}
{'loss': 0.0002, 'learning_rate': 8.90083632019116e-06, 'epoch': 2.47}
{'loss': 0.0002, 'learning_rate': 8.661887694145759e-06, 'epoch': 2.48}
{'loss': 0.0002, 'learning_rate': 8.42293906810036e-06, 'epoch': 2.49}
{'loss': 0.0909, 'learning_rate': 8.18399044205496e-06, 'epoch': 2.51}
{'loss': 0.0002, 'learning_rate': 7.945041816009559e-06, 'epoch': 2.52}
{'loss': 0.0788, 'learning_rate': 7.706093189964159e-06, 'epoch': 2.54}
{'loss': 0.0003, 'learning_rate': 7.467144563918758e-06, 'epoch': 2.55}
{'loss': 0.0002, 'learning_rate': 7.228195937873358e-06, 'epoch': 2.57}
{'loss': 0.0011, 'learning_rate': 6.989247311827957e-06, 'epoch': 2.58}
{'loss': 0.0003, 'learning_rate': 6.7502986857825566e-06, 'epoch': 2.59}
{'loss': 0.0002, 'learning_rate': 6.511350059737156e-06, 'epoch': 2.61}
{'loss': 0.0002, 'learning_rate': 6.2724014336917564e-06, 'epoch': 2.62}
{'loss': 0.0003, 'learning_rate': 6.033452807646357e-06, 'epoch': 2.64}
{'loss': 0.0003, 'learning_rate': 5.794504181600956e-06, 'epoch': 2.65}
{'loss': 0.0002, 'learning_rate': 5.555555555555556e-06, 'epoch': 2.67}
{'loss': 0.0002, 'learning_rate': 5.316606929510155e-06, 'epoch': 2.68}
{'loss': 0.0002, 'learning_rate': 5.077658303464755e-06, 'epoch': 2.7}
{'loss': 0.0002, 'learning_rate': 4.838709677419355e-06, 'epoch': 2.71}
{'loss': 0.0002, 'learning_rate': 4.599761051373955e-06, 'epoch': 2.72}
{'loss': 0.0002, 'learning_rate': 4.360812425328554e-06, 'epoch': 2.74}
{'loss': 0.0002, 'learning_rate': 4.121863799283155e-06, 'epoch': 2.75}
{'loss': 0.0002, 'learning_rate': 3.882915173237754e-06, 'epoch': 2.77}
{'loss': 0.0002, 'learning_rate': 3.643966547192354e-06, 'epoch': 2.78}
{'loss': 0.0002, 'learning_rate': 3.405017921146954e-06, 'epoch': 2.8}
{'loss': 0.0429, 'learning_rate': 3.1660692951015535e-06, 'epoch': 2.81}
{'loss': 0.0002, 'learning_rate': 2.927120669056153e-06, 'epoch': 2.82}
{'loss': 0.0002, 'learning_rate': 2.688172043010753e-06, 'epoch': 2.84}
{'loss': 0.0002, 'learning_rate': 2.449223416965353e-06, 'epoch': 2.85}
{'loss': 0.0761, 'learning_rate': 2.2102747909199524e-06, 'epoch': 2.87}
{'loss': 0.0007, 'learning_rate': 1.971326164874552e-06, 'epoch': 2.88}
{'loss': 0.0002, 'learning_rate': 1.7323775388291518e-06, 'epoch': 2.9}
{'loss': 0.0002, 'learning_rate': 1.4934289127837516e-06, 'epoch': 2.91}
{'loss': 0.0003, 'learning_rate': 1.2544802867383513e-06, 'epoch': 2.92}
{'loss': 0.0003, 'learning_rate': 1.015531660692951e-06, 'epoch': 2.94}
{'loss': 0.0144, 'learning_rate': 7.765830346475508e-07, 'epoch': 2.95}
{'loss': 0.0568, 'learning_rate': 5.376344086021506e-07, 'epoch': 2.97}
{'loss': 0.0001, 'learning_rate': 2.9868578255675034e-07, 'epoch': 2.98}
{'loss': 0.0002, 'learning_rate': 5.973715651135006e-08, 'epoch': 3.0}
```

```
  0%|          | 0/140 [00:00<?, ?it/s]
```

```
{'eval_loss': 0.026208847761154175, 'eval_accuracy': 0.9937219730941704, 'eval_runtime': 4.0835, 'eval_samples_per_second': 273.052, 'eval_steps_per_second': 34.285, 'epoch': 3.0}
{'train_runtime': 244.4781, 'train_samples_per_second': 54.717, 'train_steps_per_second': 6.847, 'train_loss': 0.0351541918909871, 'epoch': 3.0}
```

### Creating a Pipeline with the Fine-Tuned Model[​](#creating-a-pipeline-with-the-fine-tuned-model "Direct link to Creating a Pipeline with the Fine-Tuned Model")

In this section, we're going to create a pipeline that contains our fine-tuned model.

After completing the training process, our next step is to create a pipeline for inference using our fine-tuned model. This pipeline will enable us to easily make predictions with the model.

#### Setting Up the Inference Pipeline[​](#setting-up-the-inference-pipeline "Direct link to Setting Up the Inference Pipeline")

* **Pipeline Creation**: We use the `pipeline` function from the Transformers library to create an inference pipeline. This pipeline is configured for the task of text classification.
* **Model Integration**: We integrate our fine-tuned model (`trainer.model`) into the pipeline. This ensures that the pipeline uses our newly trained model for inference.
* **Configuring the Pipeline**: We set the batch size and tokenizer in the pipeline configuration. Additionally, we specify the device type, which is crucial for performance considerations.

#### Device Configuration for Different Platforms[​](#device-configuration-for-different-platforms "Direct link to Device Configuration for Different Platforms")

* **Apple Silicon (M1/M2) Devices**: For those using Apple Silicon (e.g., M1 or M2 chips), we set the device type to `"mps"` in the pipeline. This leverages Apple's Metal Performance Shaders for optimized performance on these devices.
* **Other Devices**: If you're using a device other than a MacBook Pro with Apple Silicon, you'll need to adjust the device setting to match your hardware (e.g., `"cuda"` for NVIDIA GPUs or `"cpu"` for CPU-only inference).

#### Importance of a Customized Pipeline[​](#importance-of-a-customized-pipeline "Direct link to Importance of a Customized Pipeline")

Creating a customized pipeline with our fine-tuned model allows for easy and efficient inference, tailored to our specific task and hardware. This step is vital in transitioning from model training to practical application.

In the following code block, we'll set up our pipeline with the fine-tuned model and configure it for our device.

python

```
# If you're going to run this on something other than a Macbook Pro, change the device to the applicable type. "mps" is for Apple Silicon architecture in torch.

tuned_pipeline = pipeline(
  task="text-classification",
  model=trainer.model,
  batch_size=8,
  tokenizer=tokenizer,
  device="mps",
)
```

### Validating the Fine-Tuned Model[​](#validating-the-fine-tuned-model "Direct link to Validating the Fine-Tuned Model")

In this next step, we're going to validate that our fine-tuning training was effective prior to logging the tuned model to our run.

Before finalizing our model by logging it to MLflow, it's crucial to validate its performance. This validation step ensures that the model meets our expectations and is ready for deployment.

#### Importance of Model Validation[​](#importance-of-model-validation "Direct link to Importance of Model Validation")

* **Assessing Model Performance**: We need to evaluate the model's performance on realistic scenarios to ensure it behaves as expected. This helps in identifying any issues or shortcomings in the model before it is logged and potentially deployed.
* **Avoiding Costly Redo's**: Given the large size of Transformer models and the computational resources required for training, it's essential to validate the model beforehand. If a model doesn't perform well, we wouldn't want to log the model, only to have to later delete the run and the logged artifacts.

#### Evaluating with a Test Query[​](#evaluating-with-a-test-query "Direct link to Evaluating with a Test Query")

* **Test Query**: We will pass a realistic test query to our tuned pipeline to see how the model performs. This query should be representative of the kind of input the model is expected to handle in a real-world scenario.
* **Observing the Output**: By analyzing the output of the model for this query, we can gauge its understanding and response to complex situations. This provides a practical insight into the model's capabilities post-fine-tuning.

#### Validating Before Logging to MLflow[​](#validating-before-logging-to-mlflow "Direct link to Validating Before Logging to MLflow")

* **Rationale**: The reason for this validation step is to ensure that the model we log to MLflow is of high quality and ready for further steps like deployment or sharing. Logging a poorly performing model would lead to unnecessary complications, especially considering the large size and complexity of these models.

After validating the model and ensuring satisfactory performance, we can confidently proceed to log it in MLflow, knowing it's ready for real-world applications.

In the next code block, we will run a test query through our fine-tuned model to evaluate its performance before proceeding to log it in MLflow.

python

```
# Perform a validation of our assembled pipeline that contains our fine-tuned model.
quick_check = (
  "I have a question regarding the project development timeline and allocated resources; "
  "specifically, how certain are you that John and Ringo can work together on writing this next song? "
  "Do we need to get Paul involved here, or do you truly believe, as you said, 'nah, they got this'?"
)

tuned_pipeline(quick_check)
```

```
[{'label': 'ham', 'score': 0.9985793828964233}]
```

### Model Configuration and Signature Inference[​](#model-configuration-and-signature-inference "Direct link to Model Configuration and Signature Inference")

In this next step, we generate a signature for our pipeline in preparation for logging.

After validating our model's performance, the next critical step is to prepare it for logging to MLflow. This involves setting up the model's configuration and inferring its signature, which are essential aspects of the model management process.

#### Configuring the Model for MLflow[​](#configuring-the-model-for-mlflow "Direct link to Configuring the Model for MLflow")

* **Setting Model Configuration**: We define a `model_config` dictionary to specify configuration parameters such as batch size and the device type (e.g., `"mps"` for Apple Silicon). This configuration is vital for ensuring that the model operates correctly in different environments.

#### Inferring the Model Signature[​](#inferring-the-model-signature "Direct link to Inferring the Model Signature")

* **Purpose of Signature Inference**: The model signature defines the input and output schema of the model. Inferring this signature is crucial as it helps MLflow understand the data types and shapes that the model expects and produces.
* **Using mlflow\.models.infer\_signature**: We use this function to automatically infer the model signature. We provide sample input and output data to the function, which analyzes them to determine the appropriate schema.
* **Including Model Parameters**: Along with the input and output, we also include the `model_config` in the signature. This ensures that all relevant information about how the model should be run is captured.

#### Importance of Signature Inference[​](#importance-of-signature-inference "Direct link to Importance of Signature Inference")

Inferring the signature is a key step in preparing the model for logging and future deployment. It ensures that anyone who uses the model later, either for further development or in production, has clear information about the expected data format, making the model more robust and user-friendly.

With the model configuration set and its signature inferred, we are now ready to log the model into MLflow. This will be our next step, ensuring our model is properly managed and ready for deployment.

python

```
# Define a set of parameters that we would like to be able to flexibly override at inference time, along with their default values
model_config = {"batch_size": 8}

# Infer the model signature, including a representative input, the expected output, and the parameters that we would like to be able to override at inference time.
signature = mlflow.models.infer_signature(
  ["This is a test!", "And this is also a test."],
  mlflow.transformers.generate_signature_output(
      tuned_pipeline, ["This is a test response!", "So is this."]
  ),
  params=model_config,
)
```

### Logging the Fine-Tuned Model to MLflow[​](#logging-the-fine-tuned-model-to-mlflow "Direct link to Logging the Fine-Tuned Model to MLflow")

In this next section, we're going to log our validated pipeline to the training run.

With our model configuration and signature ready, the final step in our model training and validation process is to log the model to MLflow. This step is crucial for tracking and managing the model in a systematic way.

#### Accessing the existing Run used for training[​](#accessing-the-existing-run-used-for-training "Direct link to Accessing the existing Run used for training")

* **Initiating MLflow Run**: We start a new run in MLflow using `mlflow.start_run()`. This new run is specifically for the purpose of logging the model, separate from the training run.

#### Logging the Model in MLflow[​](#logging-the-model-in-mlflow "Direct link to Logging the Model in MLflow")

* **Using mlflow\.transformers.log\_model**: We log our fine-tuned model using this function. It's specially designed for logging models from the Transformers library, making the process streamlined and efficient.

* **Specifying Model Information**: We provide several pieces of information to the logging function:

  * **transformers\_model**: The fine-tuned model pipeline.
  * **artifact\_path**: The path where the model artifacts will be stored.
  * **signature**: The inferred signature of the model, which includes input and output schemas.
  * **input\_example**: Sample inputs to give users an idea of what input the model expects.
  * **model\_config**: The configuration parameters of the model.

#### Importance of Model Logging[​](#importance-of-model-logging "Direct link to Importance of Model Logging")

Logging the model in MLflow serves multiple purposes:

* **Version Control**: It helps in keeping track of different versions of the model.
* **Model Management**: Facilitates the management of the model lifecycle, from training to deployment.
* **Reproducibility and Sharing**: Enhances reproducibility and makes it easier to share the model with others.

By logging our model in MLflow, we ensure that it is well-documented, versioned, and ready for future use, whether for further development or deployment.

python

```
# Log the pipeline to the existing training run
with mlflow.start_run(run_id=run.info.run_id):
  model_info = mlflow.transformers.log_model(
      transformers_model=tuned_pipeline,
      name="fine_tuned",
      signature=signature,
      input_example=["Pass in a string", "And have it mark as spam or not."],
      model_config=model_config,
  )
```

```
2023/11/30 12:17:11 WARNING mlflow.utils.environment: Encountered an unexpected error while inferring pip requirements (model URI: /var/folders/cd/n8n0rm2x53l_s0xv_j_xklb00000gp/T/tmp77_imuy9/model, flavor: transformers), fall back to return ['transformers==4.34.1', 'torch==2.1.0', 'torchvision==0.16.0', 'accelerate==0.21.0']. Set logging level to DEBUG to see the full traceback.
```

### Loading and Testing the Model from MLflow[​](#loading-and-testing-the-model-from-mlflow "Direct link to Loading and Testing the Model from MLflow")

After logging our fine-tuned model to MLflow, we'll now load and test it.

#### Loading the Model from MLflow[​](#loading-the-model-from-mlflow "Direct link to Loading the Model from MLflow")

* **Using mlflow\.transformers.load\_model**: We use this function to load the model stored in MLflow. This demonstrates how models can be retrieved and utilized post-training, ensuring they are accessible for future use.
* **Retrieving Model URI**: We use the `model_uri` obtained from logging the model to MLflow. This URI is the unique identifier for our logged model, allowing us to retrieve it accurately.

#### Testing the Model with Validation Text[​](#testing-the-model-with-validation-text "Direct link to Testing the Model with Validation Text")

* **Preparing Validation Text**: We use a creatively crafted text to test the model's performance. This text is designed to mimic a typical spam message, which is relevant to our model's training on spam classification.
* **Evaluating Model Output**: By passing this text through the loaded model, we can observe its performance and effectiveness in a practical scenario. This step is crucial to ensure that the model works as expected in real-world conditions.

Testing the model after loading it from MLflow is essential for several reasons:

* **Validation of Logging Process**: It confirms that the model was logged and loaded correctly.
* **Practical Performance Assessment**: Provides a real-world assessment of the model's performance, which is critical for deployment decisions.
* **Demonstrating End-to-End Workflow**: Showcases a complete workflow from training, logging, loading, to using the model, which is vital for understanding the entire model lifecycle.

In the next code block, we'll load our model from MLflow and test it with a validation text to assess its real-world performance.

python

```
# Load our saved model in the native transformers format
loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)

# Define a test example that we expect to be classified as spam
validation_text = (
  "Want to learn how to make MILLIONS with no effort? Click HERE now! See for yourself! Guaranteed to make you instantly rich! "
  "Don't miss out you could be a winner!"
)

# validate the performance of our fine-tuning
loaded(validation_text)
```

```
2023/11/30 12:17:11 INFO mlflow.transformers: 'runs:/e3260e8511c94c38aafb7124509240a4/fine_tuned' resolved as 'file:///Users/benjamin.wilson/repos/mlflow-fork/mlflow/docs/source/llms/transformers/tutorials/fine-tuning/mlruns/258758267044147956/e3260e8511c94c38aafb7124509240a4/artifacts/fine_tuned'
2023/11/30 12:17:11 WARNING mlflow.transformers: Could not specify device parameter for this pipeline type
```

```
[{'label': 'spam', 'score': 0.9873914122581482}]
```

### Conclusion: Mastering Fine-Tuning and MLflow Integration[​](#conclusion-mastering-fine-tuning-and-mlflow-integration "Direct link to Conclusion: Mastering Fine-Tuning and MLflow Integration")

Congratulations on completing this comprehensive tutorial on fine-tuning a Transformers model and integrating it with MLflow! Let's recap the essential skills and knowledge you've acquired through this journey.

#### Key Takeaways[​](#key-takeaways "Direct link to Key Takeaways")

1. **Fine-Tuning Transformers Models**: You've learned how to fine-tune a foundational model from the Transformers library. This process demonstrates the power of adapting advanced pre-trained models to specific tasks, tailoring their performance to meet unique requirements.
2. **Ease of Fine-Tuning**: We've seen firsthand how straightforward it is to fine-tune these advanced Large Language Models (LLMs). With the right tools and understanding, fine-tuning can significantly enhance a model's performance on specific tasks.
3. **Specificity in Performance**: The ability to fine-tune LLMs opens up a world of possibilities, allowing us to create models that excel in particular domains or tasks. This specificity in performance is crucial in deploying models in real-world scenarios where specialized understanding is required.

#### Integrating MLflow with Transformers[​](#integrating-mlflow-with-transformers "Direct link to Integrating MLflow with Transformers")

1. **Tracking and Managing the Fine-Tuning Process**: A significant part of this tutorial was dedicated to using MLflow for experiment tracking, model logging, and management. You've learned how MLflow simplifies these aspects, making the machine learning workflow more manageable and efficient.
2. **Benefits of MLflow in Fine-Tuning**: MLflow plays a crucial role in ensuring reproducibility, managing model versions, and streamlining the deployment process. Its integration with the Transformers fine-tuning process demonstrates the potential for synergy between advanced model training techniques and lifecycle management tools.
