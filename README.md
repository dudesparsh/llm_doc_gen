# Document Generator using LLM - Chains with RAG

This repository contains code for building a document generator using Large Language Model (LLM) with Chains using Retrieve and Generate (RAG) architecture. Developed by [Sparsh](https://www.linkedin.com/in/sparsh-jain-6bb969121/).

## Introduction

The document generator aims to create a system that can generate documents based on user inputs or prompts. It utilizes Large Language Models (LLMs) to understand and generate human-like text. Specifically, the RAG architecture is employed, which combines retrieval-based and generation-based approaches for text generation.

## Features

- Utilizes Large Language Model (LLM) for text generation
- Implements Chains model for improved coherence and consistency
- Integrates Retrieve and Generate (RAG) architecture for balanced performance
- Generates documents based on user inputs or prompts

## Current Feature

- Works for OpenAI models
- Currently use FAISS for vector Databases and Langchain for language generation
- Custom Job Description generator module/prompt is implemented
- Inputs are default, with console extendibility coming soon
- Python based running - Front-end will be added soon

## Requirements

- Python 3.x
- Required Python packages (specified in `requirements.txt`)

## Usage

1. Clone the repository:
   ```
   git clone git@github.com:dudesparsh/llm_doc_gen.git
   cd llm_doc_gen
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Set the env variables in .env file
- Currently using openai_api_key and openai_org    
  
4. Run the document generator:
   ```
   python doc_gen/DocGen.py
   ```

## Contribution

Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests.

## License

This project is licensed under the MIT License. 
