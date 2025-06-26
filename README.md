
# Agentic SelectAI

Agentic SelectAI is an advanced, agentic workflow for natural language to SQL (NL2SQL) question answering and data exploration. This example is specifically tailored for clinical/medical trial datasets. It leverages Oracle Autonomous Database 23ai, Oracle Cloud Infrastructure Generative AI, and the LangChain framework to iteratively refine user questions, generate and validate SQL, and produce concise natural language answers.

## Features

*   **Natural Language Data Query**: Ask complex clinical questions in everyday language.
*   **Medical Jargon Simplification**: Automatically simplifies clinical/medical language and translates terms to standard abbreviations.
*   **Iterative Improvement**: If the generated SQL or result is unsatisfactory, the agent refines the question and tries again.
*   **Automated SQL QA**: Checks if the SQL result matches the user’s intent using an LLM-powered QA gate.
*   **Natural Language Summarization**: Converts SQL results into clear, human-friendly answers.
*   **Visual Chat Interface**: Powered by Gradio for user-friendly interaction.

## How It Works

1.  **Question Simplification**: Medical questions are reworded and standardized using a medical term dictionary.
2.  **SQL Generation and Execution**: The system generates SQL via Oracle’s cloud AI APIs and executes it on an Oracle database.
3.  **Result QA**: The output is checked for correctness by an LLM, based on the original question and the SQL result.
4.  **Iterative Loop**: If the answer is not satisfactory after 5 iterations, the process stops. Otherwise, the question is improved and retried.
5.  **Final Answer**: A summary answer is generated in natural language from the SQL result.

## Requirements

*   Python 3.8+
*   Oracle Autonomous Database 23ai
*   OCI GenAI or any other LLM
*   `oracledb` Python library
*   `LangChain`, `LangChain Community`, `LangChain Core`
*   `dotenv`
*   `Gradio`

## Setup

1.  **Install dependencies:**
    ```bash
    pip install oracledb langchain-core langchain-community python-dotenv gradio
    ```

2.  **Set up Oracle credentials and enable Select AI**: Edit the `DB_CONFIG` dictionary in `agentic-selectai.py` with your Oracle username, password, DSN, config directory, wallet location, and wallet password.

3.  **Set up OCI Generative AI config**: Update the `model_id`, `service_endpoint`, and `compartment_id` in the `ChatOCIGenAI` instantiation.

4.  **Environment Variables**: Place your `.env` file (if needed) in the project root with any necessary environment variables.

References:
* https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/setup-oci-api-auth.htm
* https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/connect-download-wallet.html
* https://docs.oracle.com/en-us/iaas/Content/generative-ai/endpoint.htm
* https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/select-ai-get-started.html


## Usage

Run the chat interface with:

```bash
python agentic_selectai.py
```

You will be presented with a Gradio web interface where you can ask questions about your clinical trial data.

## Example Workflow

*   **User Input:**
    > “For the cohort of subjects presenting with Type 2 Diabetes Mellitus in their cardiac history, enumerate the Adverse Event Preferred Terms that resulted in a treatment discontinuation.”

*   **System Response:**
    > “For subjects who have a history of DM2, list their Subject IDs and the Adverse Event Preferred Terms, but only for those Adverse Events that led to Discontinued Treatment.”

*   **Iterative Refinement:** The system generates SQL, checks the output, and may iterate to improve the question or query.

*   **Final Answer:**
    > “The subjects with DM2 who discontinued treatment due to adverse events are: [list, summarized in plain language].”

## Customization

*   **Medical Terminology**: Expand or edit the `medical_term_dict` as needed for your domain.
*   **Prompt Templates**: Prompts for simplification, QA, and answer generation are in multi-line strings and can be customized.

## Disclaimer

This tool is intended for educational and research purposes only. It does not diagnose, treat, or advise on medical care.
