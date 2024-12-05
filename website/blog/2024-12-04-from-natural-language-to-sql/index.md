---
title: "From Natural Language to SQL: Building and Tracking a Multi-Lingual Query Engine"
description: MLflow Models from Code and MLflow Tracing applied to AI Workflows
tags:
  - pyfunc
  - mlflow
  - sql-generator
  - models-from-code
  - tracing
slug: from-natural-language-to-sql
authors:
  - hugo-carvalho
  - joana-ferreira
  - rahul-pandey
thumbnail: img/blog/from-natural-language-to-sql.png
---
If you're looking to build a Multilingual Query Engine that combines Natural Language-to-SQL generation with query execution while fully leveraging MLflow’s features, this blog post is your guide. We’ll explore how to leverage **MLflow Models from Code** to treat AI workflows as traditional ML models, enabling seamless tracking, versioning, and deployment across diverse serving infrastructures. Additionally, we’ll dive into MLflow’s **Tracing** feature, which enhances observability by tracking inputs, outputs, and metadata at every intermediate step of the AI workflow.

<!-- truncate -->

## Introduction

SQL is an essential skill needed for managing and accessing data in relational databases and many users fail to possess the necessary skills that are required to write correct SQL queries which can hinder them from utilizing data effectively. NL2SQL or Natural Language to SQL systems help in solving this problem by providing a translation from Natural Language to SQL commands allowing non-technical people to interact with data: users can just ask questions in a natural language they are comfortable speaking and these systems will assist them in returning the appropriate information.

However, there are also a number of problems that remain when creating an NL2SQL system like semantic ambiguity, schema mapping or error handling and user feedback. Therefore, it is very important that while building such systems, we must put some guardrails instead of completely relying on LLM.

In this blog post, we’ll walk you through the process of building and managing the lifecycle of a Multilingual Query Engine, encompassing both Natural Language to SQL generation and query execution.

We’ll start by demonstrating how to leverage LangGraph’s capabilities to build a dynamic AI workflow. This workflow integrates OpenAI and external data sources, such as a Vector Store and an SQLite database, to process user input, perform safety checks, query databases, and generate meaningful responses.

Throughout this post, we’ll leverage MLflow’s Models from Code feature to manage the lifecycle of the Multilingual Query Engine. This approach allows the AI workflow to be treated like a traditional ML model, enabling tracking, versioning, and deployment across various serving infrastructures.

Additionally, we’ll explore MLflow’s *Tracing* feature, designed to enhance the observability of AI workflows by tracking inputs, outputs, and metadata associated with each intermediate step of the AI Workflow. This enables easy identification of bugs and unexpected behaviors, providing greater transparency over the workflow.

# Requisites

To set up and run this project, ensure the following **Python packages** are installed:
- faiss-cpu
- langchain
- langchain-core
- langchain-openai
- langgraph
- langchain-community
- pydantic
- typing_extensions
- python-dotenv

Additionally, an **MLflow Tracking Server** is required to log and manage experiments, models, and traces effectively. For local setup, refer to the official MLflow documentation for instructions on [configuring a simple MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html).

Finally, ensure that your OpenAI API key is saved within a .env file in the project directory. This allows the application to securely access the OpenAI services required for building the AI workflow. The .env file should include a line like:

```

OPENAI_API_KEY=your_openai_api_key

```

# Multilingual Query Engine using Langraph

The Multilingual Query Engine leverages LangGraph’s advanced features to create a stateful, multi-agent, and cyclical graph architecture.
## key Langraph features:

1. **Stateful Architecture**: The engine maintains a dynamic snapshot of the graph’s execution status. This snapshot acts as a shared resource across nodes, enabling efficient decision-making and real-time updates at each node execution.

2. **Multi-Agent Design**: The AI Workflow includes multiple interactions with OpenAI and other external tools throughout the workflow.

3. **Cyclical Graph Structure**: The graph’s cyclical nature introduces a robust retry mechanism. This mechanism dynamically addresses failures by looping back to previous stages when needed, ensuring continuous graph execution. (Details of this mechanism will be discussed later.)
## AI Workflow Overview

The Multilingual Query Engine’s advanced AI workflow is composed of interconnected nodes and edges, each representing a crucial stage:

1. **Translation Node**: Converts the user’s input into English.

2. **Safety Checks**: Ensures user input is free from toxic or inappropriate content and does not contain harmful SQL commands (e.g., DELETE, DROP).

3. **Database Schema Extraction**: Retrieves the schema of the target database to understand its structure and available data.

4. **Relevancy Validation**: Validates the user’s input against the database schema to ensure alignment with the database’s capabilities.

5. **SQL Query Generation**: Generates an SQL query based on the user’s input and the current database schema.

6. **SQL Query Validation**: Executes the SQL query in a rollback-safe environment to ensure its validity before running it.

7. **Dynamic State Evaluation**: Determines the next steps based on the current state. If the SQL query validation fails, it loops back to Stage 5 to regenerate the query.

8. **Query Execution and Result Retrieval**: Executes the SQL query and returns the results if it’s a SELECT statement.

The retry mechanism is introduced in Stage 7, where the system dynamically evaluates the current graph state. Specifically, when the SQL query validation node (Stage 6) detects an issue, the state triggers a loop back to the SQL Generation node (Stage 5) for a new SQL Generation attempt (within a maximum of 3 attemps).

## Components

The Multilingual Query Engine interacts with several external components to transform Natural Language user inputs into SQL queries and execute them in a safe and robust manner. In this section, we will take a detailed look at the key AI Workflow components: OpenAI, Vector Store, SQLite Database, and SQL Generation Chain.

### OpenAI

OpenAI, more specifically gpt-4 language model plays a crucial role in multiple stages of the workflow, providing the intelligence required for:

1. **Translation**: Translates user input into English. If the text is already in English, it simply repeats the input.

2. **Safety Checks**: Analyzes user input to ensure that it does not contain toxic or inappropriate content.

3. **Relevancy Checks**: Evaluates whether the user's question is relevant given the database schema.

4. **SQL Generation**: Generates valid and executable SQL queries based on user input, SQL generation documentation, and the database schema.

Details on OpenAI implementation will be provided later on in the Node implementation section.
### FAISS Vector Store

To build an effective Natural Language to SQL engine capable of generating accurate and executable SQL queries, we leverage Langchain's FAISS Vector Store feature. This setup allows the system to search and extract SQL query generation guidelines from W3Schools SQL documents previously stored in the Vector Database, enhancing the success of SQL query generation.
#### Step 1: Load SQL Documentation
The first step in creating a FAISS Vector Store with SQL query generation guidelines is to load SQL documentation from the [W3Schools SQL page](https://www.w3schools.com/sql/) using Langchain's RecursiveUrlLoader. This tool retrieves the documentation, allowing us to use it as a knowledge base for our engine.
#### Step 2: Split the Text into Manageable Chunks
The loaded SQL documentation is a lengthy text, making it difficult to be effectively ingested by the LLM. To address this, the next step involves splitting the text into smaller, manageable chunks using Langchain's RecursiveCharacterTextSplitter. By splitting the text into chunks of 500 characters with a 50-character overlap, we ensure the AI has sufficient context while minimizing the risk of losing important information that spans across chunks. The split_text method applies this splitting process, storing the resulting pieces in a list called documents.
#### Step 3: Generate Embedding Model
The third step is to create a model that converts these chunks into embeddings (chuncks texts numerical representations). Embeddings enable the system to compare the similarity between chunks and the user's input, facilitating the retrieval of the most relevant matches for SQL query generation.
#### Step 4: Create and Store Embeddings in FAISS Vector Store
Finally, we create and store the embeddings using FAISS. The FAISS.from_texts method takes all the chunks, computes their embeddings, and stores them in a high speed searchable vector database. This searchable database allows the engine to efficiently retrieve relevant SQL guidelines, significantly improving the success rate of executable SQL query generation.

```python
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_vector_store():
    """Setup or load the vector store."""
    if not os.path.exists('data'):
        os.makedirs('data')

    vector_store_dir = 'data/vector_store'

    if os.path.exists(vector_store_dir):
        # Load the vector store from disk
        print("Loading vector store from disk...")
        vector_store = FAISS.load_local(
            vector_store_dir,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new vector store...")
        # Load SQL documentation
        url = "https://www.w3schools.com/sql/"
        loader = RecursiveUrlLoader(
            url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        documents = []
        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                documents.append({
                    "content": split,
                    "metadata": {"source": doc.metadata["source"], "chunk": i}
                })

        # Compute embeddings and create vector store
        embedding_model = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(
            [doc["content"] for doc in documents],
            embedding_model,
            metadatas=[doc["metadata"] for doc in documents]
        )

        # Save the vector store to disk
        vector_store.save_local(vector_store_dir)
        print("Vector store created and saved to disk.")

    return vector_store
```

### SQLite Database

The SQLite database is a key component of the Multilingual Query Engine, serving as the structured data repository that supports SQL query efficient generation, validation and execution by enabling:

1. **Schema Extraction**: Suplying schema information for user’s input context validation (Stage 4) and executable SQL Query Generation (Stage 5).

2. **Query Execution**: Executing SQL queries in a rollback-safe environment in Validation Stage (Stage 6) and in Query Execution Stage (Stage 8) fetching results for `SELECT` statements and committing changes for other query types.
#### SQLite Database Initialization

The database is initialized using the `setup_database` function when the AI Workflow is initialized. This process involves:

1. **Setting the SQLite Database Connection**: Establishes a connection to the SQLite database, enabling data interaction.

2. **Table Creation**: Defines and creates the necessary database tables for the AI Workflow.

3. **Data Population**: Populates the tables with sample data to support query execution and validation stages.
  

```python
import sqlite3
import os

def create_connection(db_file='data/database.db'):
    """Create a database connection to the SQLite database."""
    conn = sqlite3.connect(db_file)
    return conn

def create_tables(conn):
    """Create tables in the database."""
    cursor = conn.cursor()
    # Create Customers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Customers (
        CustomerID INTEGER PRIMARY KEY,
        CustomerName TEXT,
        ContactName TEXT,
        Address TEXT,
        City TEXT,
        PostalCode TEXT,
        Country TEXT
    )
    ''')

    # Create Orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Orders (
        OrderID INTEGER PRIMARY KEY,
        CustomerID INTEGER,
        OrderDate TEXT,
        FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID)
    )
    ''')

    # Create OrderDetails table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS OrderDetails (
        OrderDetailID INTEGER PRIMARY KEY,
        OrderID INTEGER,
        ProductID INTEGER,
        Quantity INTEGER,
        FOREIGN KEY (OrderID) REFERENCES Orders (OrderID),
        FOREIGN KEY (ProductID) REFERENCES Products (ProductID)
    )
    ''')

    # Create Products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Products (
        ProductID INTEGER PRIMARY KEY,
        ProductName TEXT,
        Price REAL
    )
    ''')

    conn.commit()

def populate_tables(conn):
    """Populate tables with sample data if they are empty."""
    cursor = conn.cursor()

    # Populate Customers table if empty
    cursor.execute('SELECT COUNT(*) FROM Customers')
    if cursor.fetchone()[0] == 0:
        customers = []
        for i in range(1, 51):
            customers.append((
                i,
                f'Customer {i}',
                f'Contact {i}',
                f'Address {i}',
                f'City {i % 10}',
                f'{10000 + i}',
                f'Country {i % 5}'
            ))
        cursor.executemany('''
        INSERT INTO Customers (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', customers)

    # Populate Products table if empty
    cursor.execute('SELECT COUNT(*) FROM Products')
    if cursor.fetchone()[0] == 0:
        products = []
        for i in range(1, 51):
            products.append((
                i,
                f'Product {i}',
                round(10 + i * 0.5, 2)
            ))
        cursor.executemany('''
        INSERT INTO Products (ProductID, ProductName, Price)
        VALUES (?, ?, ?)
        ''', products)

    # Populate Orders table if empty
    cursor.execute('SELECT COUNT(*) FROM Orders')
    if cursor.fetchone()[0] == 0:
        orders = []
        from datetime import datetime, timedelta
        base_date = datetime(2023, 1, 1)
        for i in range(1, 51):
            order_date = base_date + timedelta(days=i)
            orders.append((
                i,
                i % 50 + 1,  # CustomerID between 1 and 50
                order_date.strftime('%Y-%m-%d')
            ))
        cursor.executemany('''
        INSERT INTO Orders (OrderID, CustomerID, OrderDate)
        VALUES (?, ?, ?)
        ''', orders)

    # Populate OrderDetails table if empty
    cursor.execute('SELECT COUNT(*) FROM OrderDetails')
    if cursor.fetchone()[0] == 0:
        order_details = []
        for i in range(1, 51):
            order_details.append((
                i,
                i % 50 + 1,  # OrderID between 1 and 50
                i % 50 + 1,  # ProductID between 1 and 50
                (i % 5 + 1) * 2  # Quantity between 2 and 10
            ))
        cursor.executemany('''
        INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity)
        VALUES (?, ?, ?, ?)
        ''', order_details)

    conn.commit()

def setup_database():
    """Setup the database and return the connection."""
    db_file = 'data/database.db'
    if not os.path.exists('data'):
        os.makedirs('data')

    db_exists = os.path.exists(db_file)

    conn = create_connection(db_file)

    if not db_exists:
        print("Setting up the database...")
        create_tables(conn)
        populate_tables(conn)
    else:
        print("Database already exists. Skipping setup.")

    return conn
```

### SQL Generation Chain

The **SQL Generation Chain** (`sql_gen_chain`) is the backbone of automated SQL query generation in our workflow. This chain leverages LangChain's modular capabilities and OpenAI's advanced natural language processing to transform user questions into precise and executable SQL queries.

**Core Features**:

- **Prompt-Driven Generation**: Begins with a thoughtfully designed prompt that integrates the database schema and documentation snippets, ensuring queries are contextually accurate.

- **Structured Responses**: Delivers outputs in a predefined format, including:

- A **description** of the query's purpose.

- The corresponding **SQL code** ready for execution.

- **Adaptable and Reliable**: Uses GPT-4 for robust, consistent query generation, minimizing manual effort and errors.

This chain is a critical component in our workflow, enabling seamless integration of SQL query generation with downstream processes, ensuring accuracy, and significantly improving efficiency.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class SQLQuery(BaseModel):
    """Schema for SQL query solutions to questions."""
    description: str = Field(description="Description of the SQL query")
    sql_code: str = Field(description="The SQL code block")

def get_sql_gen_chain():
    """Set up the SQL generation chain."""
    sql_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an SQL assistant with expertise in SQL query generation. \n
Answer the user's question based on the provided documentation snippets and the database schema provided below. Ensure any SQL query you provide is valid and executable. \n
Structure your answer with a description of the query, followed by the SQL code block. Here are the documentation snippets:\n{retrieved_docs}\n\nDatabase Schema:\n
- Customers(CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)
- Orders(OrderID, CustomerID, OrderDate)
- OrderDetails(OrderDetailID, OrderID, ProductID, Quantity)
- Products(ProductID, ProductName, Price)\n
Here is the user's question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Initialize the OpenAI LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create the code generation chain
    sql_gen_chain = sql_gen_prompt | llm.with_structured_output(SQLQuery)

    return sql_gen_chain
```

### **Node Descriptions**
#### **1. Translate Input**

The `translate_input` node translates user queries into English to standardize processing, ensuring all inputs are compatible with downstream nodes.

**Purpose:** Normalize user input to English for consistent processing.

**Process:**

- Extracts the latest user input from the state.

- Uses the `ChatOpenAI` model to translate the input or verify it’s already in English.

- Updates the state with the translated input.

**Key Considerations:**

- Maintains original intent and context during translation.

- Provides accurate results even for multilingual queries.

**Examples:**

- Input: _"Quantos pedidos foram realizados em Novembro?"_

- Translated: _"How many orders were made in November?"_

- Input: _"Combien de ventes avons-nous enregistrées en France ?"_

- Translated: _"How many sales did we record in France?"_

**Code:**

```python
def translate_input(state: GraphState):
    print("---TRANSLATING INPUT---")
    messages = state["messages"]
    user_input = messages[-1][1]  # Get the latest user input

    # Translation prompt for the model
    translation_prompt = f"""
Translate the following text to English. If the text is already in English, repeat it exactly without any additional explanation.

Text:
{user_input}
"""
    # Call the OpenAI LLM to translate the text
    translated_response = llm.invoke(translation_prompt)
    translated_text = translated_response.content.strip()  # Access the 'content' attribute and strip any extra spaces
    state["translated_input"] = translated_text  # Save the translated input
    print(f"Translated Input: {translated_text}")

    return state
```

#### **2. Safety Check**

The `safety_check` node ensures user inputs are free from harmful, malicious, or potentially destructive SQL commands and inappropriate content.

**Purpose:** Filter out unsafe or harmful queries.

**Process:**

- Scans for disallowed SQL operations (*i.e.*, `CREATE`, `DELETE`, `DROP`, `INSERT`, `UPDATE`, `ALTER`, `TRUNCATE`, `EXEC`, `EXECUTE`).

- Uses the LLM to analyze the input for toxic or inappropriate content.

- Updates the state with an error flag and feedback if the query is unsafe.

**Key Considerations:**

- Rejects queries containing disallowed SQL keywords.

- Provides transparent feedback for flagged inputs.

- Ensures the query is free of offensive or harmful language.

**Examples:**

1. **Disallowed Operations:**

- **Input:** *"DROP TABLE customers;"*

- **Response:** *"Your query contains disallowed SQL operations and cannot be processed."*

- **Input:** *"SELECT * FROM orders;"*

- **Response:** *"Query allowed."*

2. **Inappropriate Content:**

- **Input:** *"Show me orders where customers have names like 'John the Idiot';"*

- **Response:** *"Your query contains inappropriate content and cannot be processed."*

- **Input:** *"Find total sales by region."*

- **Response:** *"Input is safe to process."*

**Code:**

```python
def safety_check(state: GraphState):
    print("---PERFORMING SAFETY CHECK---")
    translated_input = state["translated_input"]
    messages = state["messages"]
    error = "no"

    # List of disallowed SQL operations (e.g., DELETE, DROP)
    disallowed_operations = ['CREATE', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']
    pattern = re.compile(r'\b(' + '|'.join(disallowed_operations) + r')\b', re.IGNORECASE)

    # Check if the input contains disallowed SQL operations
    if pattern.search(translated_input):
        print("Input contains disallowed SQL operations. Halting the workflow.")
        error = "yes"
        messages += [("assistant", "Your query contains disallowed SQL operations and cannot be processed.")]
    else:
        # Check if the input contains inappropriate content
        safety_prompt = f"""
Analyze the following input for any toxic or inappropriate content.

Respond with only "safe" or "unsafe", and nothing else.

Input:
{translated_input}
"""
        safety_invoke = llm.invoke(safety_prompt)
        safety_response = safety_invoke.content.strip().lower()  # Access the 'content' attribute and strip any extra spaces and lower case

        if safety_response == "safe":
            print("Input is safe to process.")
        else:
            print("Input contains inappropriate content. Halting the workflow.")
            error = "yes"
            messages += [("assistant", "Your query contains inappropriate content and cannot be processed.")]

    state["error"] = error
    state["messages"] = messages

    return state
```

#### **3. Schema Extract**

The `schema_extract` node retrieves the schema of the connected database, forming a foundation for validating user queries.

**Purpose:** Retrieve and store database schema information.

**Process:**

- Queries database metadata to fetch table names and column details.

- Formats the schema as a string and saves it in the state.

**Key Considerations:**

- Dynamically adapts to the current database structure.

- Supports detailed extraction for multiple tables and columns.

**Examples:**

- Input: Request for schema extraction.

- Schema Output:

    - Customers(CustomerID (INTEGER), CustomerName (TEXT), ContactName (TEXT), Address (TEXT), City (TEXT), PostalCode (TEXT), Country (TEXT))

    - Orders(OrderID (INTEGER), CustomerID (INTEGER), OrderDate (TEXT))

    - OrderDetails(OrderDetailID (INTEGER), OrderID (INTEGER), ProductID (INTEGER), Quantity (INTEGER))

    - Products(ProductID (INTEGER), ProductName (TEXT), Price (REAL))

**Code:**

```python
def schema_extract(state: GraphState):
    print("---EXTRACTING DATABASE SCHEMA---")
    # Extract the schema from the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_details = []

    # Loop through each table and retrieve column information
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_defs = ', '.join([f"{col[1]} ({col[2]})" for col in columns])
        schema_details.append(f"- {table_name}({column_defs})")

    # Save the schema in the state
    database_schema = '\n'.join(schema_details)
    state["database_schema"] = database_schema
    print(f"Database Schema:\n{database_schema}")

    return state
```

#### **4. Context Check**

The `context_check` node validates the user query against the extracted database schema to ensure relevance.

**Purpose:** Confirm that the query is actionable within the database’s schema.

**Process:**

- Compares the translated input with the schema using the LLM.

- Flags the query as irrelevant if it doesn’t align with the schema.

**Key Considerations:**

- Prevents resource waste on irrelevant queries.

- Provides user feedback for reformulation if needed.

**Examples:**

- Input: _"What is the average order value?"_

- Response: Input is relevant to the database schema.

- Input: _"Show me data from the inventory table."_

- Response: _"Your question is not related to the database and cannot be processed."_

**Code:**

```python
def context_check(state: GraphState):
    print("---PERFORMING CONTEXT CHECK---")
    translated_input = state["translated_input"]
    messages = state["messages"]
    error = "no"
    database_schema = state["database_schema"]  # Get the schema from the state

    # Use the LLM to determine if the input is relevant to the database schema
    context_prompt = f"""
Determine whether the following user input is a question that can be answered using the database schema provided below.

Respond with only "relevant" if the input is relevant to the database schema, or "irrelevant" if it is not.

User Input:
{translated_input}

Database Schema:
{database_schema}
"""
    # Call the LLM for context check
    llm_invoke = llm.invoke(context_prompt)
    llm_response = llm_invoke.content.strip().lower()  # Access the 'content' attribute and strip any extra spaces and lower case

    if llm_response == "relevant":
        print("Input is relevant to the database schema.")
    else:
        print("Input is not relevant. Halting the workflow.")
        error = "yes"
        messages += [("assistant", "Your question is not related to the database and cannot be processed.")]

    state["error"] = error
    state["messages"] = messages

    return state
```

#### **5. Generate**

The `generate` node constructs a SQL query based on the user input and the database schema.

**Purpose:** Generate SQL queries from natural language input.

**Process:**

- Retrieves relevant documents from the vector store for contextual information.

- Uses a pre-defined SQL generation chain to create a query aligned with the user’s intent.

- Updates the state with the generated SQL query and description.

**Key Considerations:**

- Handles complex queries by leveraging schema context.

- Iteratively retries query generation if errors occur.

**Examples:**

- Input: _"Find total sales."_

- Generated SQL: _"SELECT SUM(Products.Price * OrderDetails.Quantity) AS TotalSales FROM OrderDetails LEFT JOIN Products ON OrderDetails.ProductID = Products.ProductID;"_

- Input: _"List all customers in New York."_

- Generated SQL: _"SELECT name FROM customers WHERE location = 'New York';"_

**Code:**

```python
def generate(state: GraphState):
    print("---GENERATING SQL QUERY---")
    messages = state["messages"]
    iterations = state["iterations"]
    translated_input = state["translated_input"]

    # Retrieve relevant documents from the vector store using the translated input
    docs = vector_store.similarity_search(translated_input, k=4)
    retrieved_docs = "\n\n".join([doc.page_content for doc in docs])

    # Generate the SQL query using the chain
    sql_solution = sql_gen_chain.invoke(
        {
            "retrieved_docs": retrieved_docs,
            "messages": [("user", translated_input)],
        }
    )

    # Save the generated SQL query in the state
    messages += [
        (
            "assistant",
            f"{sql_solution.description}\nSQL Query:\n{sql_solution.sql_code}",
        )
    ]
    iterations += 1

    state["generation"] = sql_solution
    state["messages"] = messages
    state["iterations"] = iterations

    return state
```

#### **6. SQL Check**

The `sql_check` node validates the generated SQL query for safety and integrity before execution.

**Purpose:** Ensure the SQL query adheres to safety and syntactical standards.

**Process:**

- Executes the query within a transactional savepoint to test its validity.

- Rolls back any changes after validation.

- Flags errors and updates the state if validation fails.

**Key Considerations:**

- Detects potentially destructive operations.

- Provides detailed feedback on validation errors.

**Examples:**

- Input SQL: _"SELECT name FROM customers WHERE city = 'New York';"_

- Validation: Query is valid.

- Input SQL: _"SELECT MONTH(date) AS month, SUM(total) AS total_sales FROM orders GROUP BY MONTH(date);"_

- Response: _"Your SQL query failed to execute: no such function: MONTH."_

**Code:**

```python
def sql_check(state: GraphState):
    print("---VALIDATING SQL QUERY---")
    messages = state["messages"]
    sql_solution = state["generation"]
    error = "no"

    sql_code = sql_solution.sql_code.strip()

    try:
        # Start a savepoint for the transaction
        conn.execute('SAVEPOINT sql_check;')
        # Attempt to execute the SQL query
        cursor.execute(sql_code)
        # Roll back to the savepoint to undo any changes
        conn.execute('ROLLBACK TO sql_check;')
        print("---SQL QUERY VALIDATION: SUCCESS---")
    except Exception as e:
        # Roll back in case of error
        conn.execute('ROLLBACK TO sql_check;')
        print("---SQL QUERY VALIDATION: FAILED---")
        print(f"Error: {e}")
        messages += [("user", f"Your SQL query failed to execute: {e}")]
        error = "yes"

    state["error"] = error

    return state
```
#### **7. Run Query**

The `run_query` node executes the validated SQL query and retrieves the results.

**Purpose:** Execute the SQL query and provide the output.

**Process:**

- Connects to the database to run the query.

- Fetches results and updates the state with the retrieved data.

**Key Considerations:**

- Implements robust error handling for execution issues.

- Ensures results are in a usable format for further analysis or reporting.

**Examples:**

- Input SQL: _"SELECT COUNT(*) FROM Customers WHERE City = 'New York';"_

- Query Result: _"(0,)"_

- Input SQL: _"SELECT SUM(Products.Price * OrderDetails.Quantity) AS TotalSales FROM OrderDetails LEFT JOIN Products ON OrderDetails.ProductID = Products.ProductID;"*

- Query Result: _"(6925.0,)"_

**Code:**

```python
def run_query(state: GraphState):
    print("---RUNNING SQL QUERY---")
    sql_solution = state["generation"]
    sql_code = sql_solution.sql_code.strip()
    results = None
    no_records_found = False  # Flag to indicate no records found

    try:
        # Execute the SQL query
        cursor.execute(sql_code)
        # For SELECT queries, fetch and store results
        if sql_code.upper().startswith("SELECT"):
            results = cursor.fetchall()
            if not results:
                no_records_found = True
                print("---SQL QUERY EXECUTION: SUCCESS---")
                print("No records found.")
            else:
                print("---SQL QUERY EXECUTION: SUCCESS---")
        else:
            # For non-SELECT queries, commit the changes
            conn.commit()
            print("---SQL QUERY EXECUTION: SUCCESS---")
    except Exception as e:
        print("---SQL QUERY EXECUTION: FAILED---")
        print(f"Error: {e}")

    # Store the results and no_records_found flag in the state
    state["results"] = results
    state["no_records_found"] = no_records_found
    return state
```

The schema above provides a representation of the LangGraph Nodes and Edges:

![graph_representation](graph.png)

# Logging the Model in MLFlow

Now that we have built a Multi-Lingual Query Engine using LangGraph, we are ready to log the model using MLflow’s [ Model from Code](https://mlflow.org/blog/models_from_code). This approach, where we log the code that represents the model, contrasts with object-based logging, where a model object is created, serialized, and logged as a pickle or JSON object.

## Step 1: Create our Model-From-Code File

So far, we have defined the get_workflow function in a file named workflow.py. In this step, we will create a new file, sql_model.py, which introduces the SQLGenerator class. This script will:

1. Import the get_workflow function from workflow.py.
2. Define the SQLGenerator class as a PythonModel, including a predict method that utilizes the get_workflow function to initialize the LangGraph workflow.
3. Use mlflow.models.set_model to designate the SQLGenerator PythonModel class as the model of interest for MLflow.

```python
import mlflow
from definitions import REMOTE_SERVER_URI
from workflow import get_workflow

mlflow.set_tracking_uri(REMOTE_SERVER_URI)


class SQLGenerator(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return get_workflow(
            model_input["conn"], model_input["cursor"], model_input["vector_store"]
        )


mlflow.models.set_model(SQLGenerator())
```

## Step 2: Log with "Model from Code" feature

With the SQLGenerator custom Python model defined, the next step is to log it using MLflow's "Model from Code" feature. This involves using the log model standard API specifying the path to the sql_model.py script and using the code_paths parameter to include workflow.py as a dependency. This approach ensures that all necessary code files are available when the model is loaded in a different environment or on another machine.

```python
import mlflow
from definitions import (
    EXPERIMENT_NAME,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    REMOTE_SERVER_URI,
)
from mlflow import MlflowClient

client = MlflowClient(tracking_uri=REMOTE_SERVER_URI)

mlflow.set_tracking_uri(REMOTE_SERVER_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    logged_model_info = mlflow.pyfunc.log_model(
        python_model="sql_model.py",
        artifact_path="sql_generator",
        registered_model_name=REGISTERED_MODEL_NAME,
        code_paths=["workflow.py"],
    )

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME, MODEL_ALIAS, logged_model_info.registered_model_version
)

print(f"MLflow Run: {logged_model_info.run_id}")
print(f"Model URI: {logged_model_info.model_uri}")
```

In the MLflow UI, the stored model includes both the `sql_model.py` and `workflow.py` scripts as artifacts within the run. This logging from code feature not only records the model's parameters and metrics but also captures the code defining its functionality. This ensures observability, seamless tracking, and straightforward debugging directly through the UI.

![model_as_code_artifact](model_as_code.png)

# Use the Logged Multi-Lingual Query Engine in main.py

After logging the model, it can be loaded back from the MLflow tracking server using the model URI and the standard mlflow.pyfunc.load_model API. When the model is loaded, the workflow.py script will be executed along with the sql_model.py script, ensuring that the get_workflow function is available when the predict method is called.

Executing the code below, we demonstrate that our Multi-Lingual Query Engine is able to perform Natural Language to SQL generation and query execution.

```python
import os

import mlflow
from database import setup_database
from definitions import (
    EXPERIMENT_NAME,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    REMOTE_SERVER_URI,
)
from dotenv import load_dotenv
from vector_store import setup_vector_store

mlflow.set_tracking_uri(REMOTE_SERVER_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.langchain.autolog()


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Access secrets using os.getenv
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Setup database and vector store
    conn = setup_database()
    cursor = conn.cursor()
    vector_store = setup_vector_store()

    # Load the model
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    model_input = {"conn": conn, "cursor": cursor, "vector_store": vector_store}
    app = model.predict(model_input)

    # save image
    app.get_graph().draw_mermaid_png(
        output_file_path="sql_agent_with_safety_checks.png"
    )

    # Example user interaction
    print("Welcome to the SQL Assistant!")
    while True:
        question = input("\nEnter your SQL question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        # Initialize the state with all required keys
        initial_state = {
            "messages": [("user", question)],
            "iterations": 0,
            "error": "",
            "results": None,
            "generation": None,
            "no_records_found": False,
            "translated_input": "",  # Initialize translated_input
        }

        solution = app.invoke(initial_state)

        # Check if an error was set during the safety check
        if solution["error"] == "yes":
            print("\nAssistant Message:\n")
            print(solution["messages"][-1][1])  # Display the assistant's message
            continue  # Skip to the next iteration

        # Extract the generated SQL query from solution["generation"]
        sql_query = solution["generation"].sql_code
        print("\nGenerated SQL Query:\n")
        print(sql_query)

        # Extract and display the query results
        if solution.get("no_records_found"):
            print("\nNo records found matching your query.")
        elif "results" in solution and solution["results"] is not None:
            print("\nQuery Results:\n")
            for row in solution["results"]:
                print(row)
        else:
            print("\nNo results returned or query did not execute successfully.")

    print("Goodbye!")


if __name__ == "__main__":
    main()

```

# Project File Structure

The Project follows a simple file structure:

<img src="file_structure.png" alt="file_structure" width="300">

# MLFlow Tracing

[MLflow Automated Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html) provides fully automated integrations with various GenAI libraries such as LangChain, OpenAI, LlamaIndex, DSPy, and AutoGen. Since our AI Workflow is built using LangGraph, we can activate automated LangChain tracing by enabling `mlflow.langchain.autolog()`.

With LangChain autologging, traces are automatically logged to the active MLflow experiment whenever invocation APIs are called on chains. This seamless integration ensures that every interaction is captured for monitoring and analysis.

## Viewing Traces in MLflow

Traces can be easily accessed by navigating to the MLflow experiment of interest and clicking on the "Tracing" tab. Once inside, selecting a specific trace provides detailed execution information.

Each trace includes:

1. **Execution Graphs**: Visualizations of the workflow steps.
2. **Inputs and Outputs**: Detailed logs of data processed at each step.

This granular visibility enables developers to debug and optimize their workflows effectively.

By leveraging MLflow tracing, we ensure that our Multi-Lingual Query Engine remains transparent, auditable, and scalable.

![mlflow_tracing_gif](mlflow_trace.gif)

## Conclusion

Throughout this post, we explored the process of building and managing a Multi-Lingual Query Engine using LangGraph and MLflow. By integrating LangGraph’s dynamic AI workflows with MLflow’s robust lifecycle management and tracing features, we’ve created a system that not only delivers accurate and efficient Natural Language to SQL generation and execution but is also transparent and scalable.