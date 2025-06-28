"""
This script implements a multi-agent AI system using LangGraph to answer natural
language questions about medical trial data stored in an Oracle Database.

The system follows a refined query process:
1.  An initial question is sent to Oracle's Select AI to generate and execute a SQL query.
2.  A "QA Agent" (LLM) evaluates if the query result correctly answers the question.
3.  If the result is unsatisfactory, a "Refiner Agent" (LLM) improves the original
    question based on the feedback and a medical term dictionary. The process repeats.
4.  Once the QA Agent passes the result, a final agent converts the structured
    JSON data into a human-readable, natural language answer.
5.  The entire application is exposed through a Gradio chat interface.
"""

import os
from typing import List

import gradio as gr
import oracledb
from dotenv import load_dotenv
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Load environment variables from a .env file
load_dotenv()

# --- CONSTANTS ---

# A dictionary for translating common medical terms to their standard abbreviations.
MEDICAL_TERM_DICT = """
Patient = Subject
Coronary Artery Bypass Graft = CABG
Type 1 Diabetes Mellitus = DM1
Type 2 Diabetes Mellitus = DM2
Myocardial Infarction = MI
Peripheral Artery Disease = PAD
Anatomical Therapeutic Chemical = ATC
concomitant medication = drugs
Abdominal Aortic Aneurysm = AAA
Coronary Artery Disease = CAD
Systolic Blood Pressure = SYSBP
Diastolic Blood Pressure = DIABP
Electrocardiogram = ECGP
Initial Point of Reference Check = baseline
Initial visit = baseline visit"""

# Prompt template for the "Refiner Agent" to improve the user's question.
IMPROVE_QUESTION_TEMPLATE = """You are a highly skilled medical language simplification expert. Your task is to take complex, medically worded clinical questions and rephrase them into simpler, more direct natural language questions, **using a provided medical terminology translation dictionary to replace medical terms with their standard medical abbreviations.**

**The Goal:**

The purpose of this re-wording is to make the questions easier for a separate Natural Language to SQL (NL2SQL) Language Model to understand and translate into SQL queries. The NL2SQL LLM is not specifically trained on complex medical terminology, so simplification is crucial. You will be provided feedback on issues with the question and its SQL result, address these issues in your rephrased question by targeting the feedback.

**Input:**

You will receive clinical questions that are phrased using medical jargon and terminology common in healthcare or clinical research settings. These questions might include:

*   Formal medical terms (e.g., "coronary artery disease," "diabetes mellitus," "adverse events").
*   Clinical phrasing and sentence structures.
*   Implicit understanding of medical data context.
*   **Medical abbreviations (which you must translate using the provided dictionary).**

**You will also be provided with a Medical Terminology Translation Dictionary to help you understand and translate medical terms into standard medical abbreviations.**


**Output:**

Your goal is to output a simpler, more direct natural language question that:

*   **Retains the original meaning and intent** of the medical question completely. Do not change the question's objective, only its phrasing.
*   Uses **common, everyday vocabulary** where possible, AND **standard medical abbreviations for medical terms, using the provided dictionary.**
*   Employs **simpler sentence structures** that are easier to parse.
*   Presents the request in a **more direct and explicit** manner.
*   **Avoids ambiguity** and ensures the question is clear and concise.
*   **Crucially, MUST NOT contain any database table names, column names, or technical database terms.** The reworded question should be understandable to someone without database knowledge, **but it SHOULD use standard medical abbreviations.**

**Examples:**

Here are some examples of medically worded questions and their desired reworded simpler versions:

**Example 1:**

*   **Medically Worded Question:** "Identify all subjects with a documented history of coronary artery disease that came in for an initial point of reference check"
*   **Reworded Question:** "List patient IDs who have CAD and visited for baseline."

**Example 2:**

*   **Medically Worded Question:** "For the cohort of subjects presenting with Type 2 Diabetes Mellitus in their cardiac history, enumerate the Adverse Event Preferred Terms that resulted in a treatment discontinuation, and identify the corresponding subjects."
*   **Reworded Question:** "For subjects who have a history of DM2, list their Subject IDs and the Adverse Event Preferred Terms, but only for those Adverse Events that led to Discontinued Treatment."

**Example 3:**

*   **Medically Worded Question:** "Retrieve the count of subjects currently administered Aspirin as concomitant medication, categorized by visit type."
*   **Reworded Question:** "Count how many subjects are taking the drug Aspirin, grouped by visit name."

**Example 4:**

*   **Medically Worded Question:** "Investigate the proportion of female subjects experiencing Headache as an adverse event, presented as a percentage of all female subjects."
*   **Reworded Question:** "What percentage of female subjects experienced Headache as an AE?"

**Instructions and Constraints:**

*   **Focus on Simplification and Abbreviation Translation:** Your primary goal is to simplify the language AND translate medical terms to standard abbreviations using the provided dictionary.
*   **Use the Medical Terminology Translation Dictionary:** Refer to the dictionary to translate medical terms to standard medical abbreviations.
*   **Avoid Technical Database Terms:** Do not include any database table names, column names, SQL keywords, or other technical database jargon.  **It's OK to use medical abbreviations in the reworded question.**
*   **Maintain Clarity:** Ensure the reworded question is clear, concise, and easy to understand for someone without medical or technical expertise.
*   **Preserve Intent:** The reworded question must still ask for the same information as the original medical question.
*   **Be Direct:** Rephrase questions to be more direct and action-oriented (e.g., "List," "Count," "Sum" "What percentage", "Group by").
*   **Drastically Change Approach (After 4 Turns):** After the question history has 4 entries, you must try a drastically different approach to rephrasing the question to explore different simplification strategies.
*   **Ensure you do not produce any of the questions in the question history or the provided question itself.**

**Example Input:**


"What is the average systolic blood pressure for subjects at baseline visit who are in the Cardiovex treatment arm?"

Generated code
**Example Expected Output:**
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

"What is the average SYSBP for patients at their first visit who are receiving Cardiovex treatment?"

Generated code
**Now, please reword the following medically worded clinical question into a simpler, more direct natural language question, remembering to avoid any table or column names and maintain the original intent:**

Question: {question}
Feedback: {feedback}
Question History: {questionhistory}

Medical Terminology Translation Dictionary:
{medical_term_dictionary}
"""

# Prompt template for the "QA Gate Agent" to validate the SQL result.
GATE_PROMPT_TEMPLATE = """You are a highly skilled SQL Quality Assurance (QA) expert. Your task is to evaluate if a provided SQL query accurately answers a user's natural language question, based on the question and the **actual JSON formatted result** of executing that query.

**Important:** You **cannot execute any SQL queries** and **will not be provided with the SQL query itself**. You must determine correctness based *only* on the *text* of the user question and the **JSON formatted result** of executing the SQL query. You do not have access to any database schema, table names, or column names beyond what is provided in the question and JSON result.

**Input:**

You will receive two pieces of information:

1.  **User Question (Natural Language):** The original question asked by a user in natural language.
2.  **SQL Query Result (JSON):** The *actual* result of executing the SQL query, provided in JSON format.

**Output:**

Your task is to determine if the **SQL Query Result (JSON)** is correct and **effectively answers** the **User Question**, based *only* on the **User Question** and the **SQL Query Result (JSON)**.

You must output one of two responses:

*   **"Pass"**: If you judge that the SQL query result (JSON) is relevant and plausibly answers the user's question, focusing on the *intent* of the question and the *usefulness* of the result.
*   **"Fail: [Reason]"**: If you judge that the SQL query result (JSON) is incorrect, does not fully answer the question, or is logically flawed *in terms of relevance to the question*. In this case, provide a concise reason for the failure, focusing on why the *result data* is not a good answer to the *question asked*.

**Evaluation Criteria for "Correctness":**

A "correct" SQL Query Result (JSON) is one that:

*   **Answers the User Question's Intent:** The data in the JSON result should provide the information the user is asking for, understanding that user questions might be naturally phrased and imply context.
*   **Plausible and Relevant Result:** The *data* in the JSON result should be a reasonable and relevant answer to the user's question. Consider:
    *   Does the *structure* of the JSON result (e.g., columns, grouping) provide a sensible format for answering the question?
    *   Does the *type of data* present in the JSON values seem like the kind of information the user is requesting?
    *   **Based on the question, is the JSON result generally *useful* and *on-topic*, even if not absolutely perfect?**  Focus on relevance and intent, not strict technical perfection of the (unseen) SQL.

**Examples of Evaluation Scenarios:**

**Example 1 (Pass - Relaxed Check):**

*   **User Question:** "List patient IDs and their visit names for all patients who have Coronary Artery Disease."
*   **SQL Query Result (JSON):**
    {#
    ```json
    [
      {"Patient_ID": "CVX-PH3C-100"},
      {"Patient_ID": "CVX-PH3C-170"},
      {"Patient_ID": "CVX-PH3C-232"},
      {"Patient_ID": "CVX-PH3C-233"}
    ]
    ```
    #}
*   **Your Output:** `Pass` (Reasoning: Although visit names are missing from the result, the JSON provides a list of Patient IDs which is *partially* relevant and useful for the question.  Given the question's intent is to identify patients with CAD, this result is deemed acceptable and 'passes' the QA.)

**Example 2 (Fail - Not Relevant):**

*   **User Question:** "List patient IDs and their visit names for all patients who have Coronary Artery Disease."
*   **SQL Query Result (JSON):**
    {#
    ```json
    [
      {"ITEMSETID": "110880", "ITEMSETINDEX": 4},
      {"ITEMSETID": "127491", "ITEMSETINDEX": 1},
      {"ITEMSETID": "150000", "ITEMSETINDEX": 4}
    ]
    ```
    #}
*   **Your Output:** `Fail: Irrelevant data. The JSON result contains ITEMSETID and ITEMSETINDEX, which are not related to patient IDs or visit names, and doesn't answer the question about patients with Coronary Artery Disease.`

**Example 3 (Fail - Missing Information):**

*   **User Question:** "How many adverse events of 'Migraine' were recorded, broken down by the severity of the event?"
*   **SQL Query Result (JSON):**
    {#
    ```json
    [
      {"AESEV": "Mild"},
      {"AESEV": "Moderate"},
      {"AESEV": "Severe"}
    ]
    ```
    #}
*   **Your Output:** `Fail: Result incomplete. The JSON result lists severity levels but doesn't provide counts for each severity, failing to "break down by severity" as requested.`


**Task:**

Please evaluate if the provided **SQL Query Result (JSON)** is correct and effectively answers the **User Question**, based on the criteria above. Output either "Pass" or "Fail: [Reason]".
Question: {{ question }}
Simulated SQL Result: {{ result }}
"""

# Prompt template for the "Formatter Agent" to convert final JSON to natural language.
RESULT_TO_NL_TEMPLATE = """You are an expert in generating clear and concise natural language answers from structured data. Your task is to take a user's original question, the SQL query that was intended to answer it, and the JSON formatted result of executing that SQL query, and produce a human-readable answer to the user's question.

**Input:**

You will receive three pieces of information:

1.  **User Question (Natural Language):** The original question asked by a user in natural language.
2.  **Generated SQL Query:** The SQL query produced by an NL2SQL Language Model (for context, but you don't need to understand SQL in detail).
3.  **SQL Query Result (JSON):** The result of executing the SQL query, provided in JSON format. This is the data you will use to construct your answer.

**Output:**

Your goal is to generate a concise and informative natural language answer that directly answers the **User Question** using the data provided in the **SQL Query Result (JSON)**.

**Key Requirements for the Natural Language Answer:**

*   **Directly Answer the User Question:**  The answer must clearly and directly address the question asked by the user.
*   **Use Information from JSON Result:**  Your answer MUST be based *only* on the data provided in the JSON result. Do not make assumptions or use outside information.
*   **Clarity and Conciseness:**  The answer should be easy to understand, using clear and simple language. Be concise and avoid unnecessary jargon.
*   **Natural Language Format:**  Output a natural-sounding sentence or paragraph, not code, JSON, or table formats.
*   **Context from SQL (Optional but Helpful):** While not strictly necessary to *understand* SQL, reviewing the **Generated SQL Query** can provide context about what data was intended to be retrieved, which might help you formulate a more relevant and accurate answer.

**Example Scenarios:**

**Example 1:**

*   **User Question:** "List patient IDs and their visit names for all patients who have Coronary Artery Disease."
*   **Generated SQL Query:** `SELECT SUBJECTID, VISITNAME FROM CHD_DATA_TABLE WHERE CHDCAD = 'Yes';`
*   **SQL Query Result (JSON):**
{#
[
  {"SUBJECTID": "CVX-PH3C-100", "VISITNAME": "Baseline"},
  {"SUBJECTID": "CVX-PH3C-170", "VISITNAME": "Follow-up"},
  {"SUBJECTID": "CVX-PH3C-232", "VISITNAME": "Baseline"},
  {"SUBJECTID": "CVX-PH3C-233", "VISITNAME": "Baseline"}
]
#}
*   **Your Output:** "The patients with Coronary Artery Disease are: CVX-PH3C-100 at Baseline visit, CVX-PH3C-170 at Follow-up visit, CVX-PH3C-232 at Baseline visit, and CVX-PH3C-233 at Baseline visit."

**Example 2:**

*   **User Question:** "How many adverse events of 'Migraine' were recorded, broken down by the severity of the event?"
*   **Generated SQL Query:** `SELECT AESEV, COUNT(*) FROM AE_DATA_TABLE WHERE AETERM = 'Migraine' GROUP BY AESEV;`
*   **SQL Query Result (JSON):**
{#
[
  {"AESEV": "Mild", "COUNT(*)": 5},
  {"AESEV": "Moderate", "COUNT(*)": 2},
  {"AESEV": "Severe", "COUNT(*)": 1}
]
#}
*   **Your Output:** "There were 5 mild migraine adverse events, 2 moderate migraine events, and 1 severe migraine event recorded."

**Example 3:**

*   **User Question:** "What percentage of female subjects experienced 'Headache' as an adverse event?"
*   **Generated SQL Query:** `SELECT (COUNT(CASE WHEN DM.SEX = 'Female' AND AE.PREF = 'Headache' THEN 1 END) * 100.0 / COUNT(DISTINCT DM.SUBJECTID)) AS Percentage_Female_Headache FROM DM_DATA_TABLE DM JOIN AE_DATA_TABLE AE ON DM.SUBJECTID = AE.SUBJECTID;`
*   **SQL Query Result (JSON):**
{#
[
  {"Percentage_Female_Headache": "7.5"}
]
#}
*   **Your Output:** "Approximately 7.5pct of female subjects experienced headache as an adverse event."

**Task:**

Please generate a natural language answer to the **User Question** using the provided **SQL Query Result (JSON)**, considering the **Generated SQL Query** for context.

User Question: {{ question }}
Generated SQL Query: {{ sql }}
SQL Query Result: {{ result }}
"""

MAX_RETRIES = 5
# A global counter to prevent infinite loops in the refinement process.
retry_counter = 0


# --- CONFIGURATION ---

def setup_llm() -> ChatOCIGenAI:
    """Initializes and returns the OCI Generative AI chat model."""
    return ChatOCIGenAI(
        model_id=os.environ.get("OCI_MODEL_ID"),
        service_endpoint=os.environ.get("OCI_SERVICE_ENDPOINT"),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
    )


def setup_database_connection() -> oracledb.Connection:
    """
    Reads database configuration from environment variables and
    returns an Oracle database connection object.
    """
    db_config = {
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "dsn": os.environ.get("DB_DSN"),
        "config_dir": os.environ.get("DB_CONFIG_DIR"),
        "wallet_location": os.environ.get("DB_WALLET_LOCATION"),
        "wallet_password": os.environ.get("DB_WALLET_PASSWORD"),
    }
    print("🔌 Connecting to Oracle Database...")
    connection = oracledb.connect(
        user=db_config["user"],
        password=db_config["password"],
        dsn=db_config["dsn"],
        config_dir=db_config["config_dir"],
        wallet_location=db_config["wallet_location"],
        wallet_password=db_config["wallet_password"],
    )
    print("✅ Database connection successful.")
    return connection


# --- LANGGRAPH STATE AND NODES ---

class State(TypedDict):
    """Defines the state structure for the LangGraph workflow."""
    questionlatest: str
    sqllatest: str
    resultlatest: str
    questionhistory: List[str]
    sqlhistory: List[str]
    resulthistory: List[str]
    feedback: str
    nl: str


def selectai(state: State, connection: oracledb.Connection) -> dict:
    """
    Node that uses Oracle Select AI to generate and execute a SQL query.

    Args:
        state: The current state of the workflow.
        connection: The Oracle database connection object.

    Returns:
        A dictionary updating the state with the latest SQL and result.
    """
    print(f"💿 [SelectAI] Received question: {state['questionlatest']}")

    query = """
        SELECT DBMS_CLOUD_AI.GENERATE(
            prompt       => :prompt,
            profile_name => 'OCI_GENAI',
            action       => :action
        )
        FROM dual
    """
    
    # First, get the SQL explanation
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, {'prompt': state['questionlatest'], 'action': 'explainsql'})
            result = cursor.fetchone()
            if result and isinstance(result[0], oracledb.LOB):
                sql_explanation = result[0].read()
                print(f"💿 [SelectAI] Generated SQL: {sql_explanation}\n")
                state['sqllatest'] = sql_explanation[:3000]
            else:
                state['sqllatest'] = "NONE"
    except Exception as e:
        print(f"🚨 [SelectAI] Error generating SQL: {e}")
        return {"sqllatest": "NONE", "resultlatest": "NONE"}

    # Second, run the SQL to get the result
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, {'prompt': state['questionlatest'], 'action': 'runsql'})
            result = cursor.fetchone()
            if result and isinstance(result[0], oracledb.LOB):
                text_result = result[0].read()
                print(f"💿 [SelectAI] Query Result (JSON): {text_result}\n")
                state['resultlatest'] = text_result[:3000]
            else:
                state['resultlatest'] = "NONE"
    except Exception as e:
        print(f"🚨 [SelectAI] Error executing SQL: {e}")
        state['resultlatest'] = "NONE"

    return {"sqllatest": state['sqllatest'], "resultlatest": state['resultlatest']}


def improve_question(state: State, llm: ChatOCIGenAI) -> dict:
    """
    Node that calls an LLM to refine the question based on feedback.

    Args:
        state: The current state of the workflow.
        llm: The language model instance.

    Returns:
        A dictionary updating the state with the new, improved question.
    """
    print("👨🏼‍🔧 [Refiner Agent] Improving question based on feedback...")
    prompt_template = PromptTemplate(
        input_variables=["question", "questionhistory", "feedback", "medical_term_dictionary"],
        template=IMPROVE_QUESTION_TEMPLATE
    )
    prompt_string = prompt_template.format(
        question=state['questionlatest'],
        questionhistory=state['questionhistory'],
        feedback=state['feedback'],
        medical_term_dictionary=MEDICAL_TERM_DICT
    )
    
    messages = [HumanMessage(content=prompt_string)]
    response = llm.invoke(messages)
    improved_question = response.content

    state['questionhistory'].append(state['questionlatest'])
    state['questionlatest'] = improved_question

    print(f"👨🏼‍🔧 [Refiner Agent] New improved question: {improved_question}\n")
    return {"questionlatest": improved_question, "questionhistory": state['questionhistory']}


def check_selectai_output(state: State, llm: ChatOCIGenAI) -> str:
    """
    Conditional edge. Evaluates the SQL result and decides the next step.

    Args:
        state: The current state of the workflow.
        llm: The language model instance.

    Returns:
        'Pass' to proceed to the final answer generation, or 'Fail' to refine the question.
    """
    global retry_counter
    retry_counter += 1
    print(f"👷🏽‍♂️ [QA Agent] Checking result quality (Attempt {retry_counter}/{MAX_RETRIES})...")
    
    prompt_template = PromptTemplate(
        input_variables=["question", "result"],
        template=GATE_PROMPT_TEMPLATE,
        template_format="jinja2"
    )
    prompt_string = prompt_template.format(question=state['questionlatest'], result=state['resultlatest'])
    messages = [HumanMessage(content=prompt_string)]
    response = llm.invoke(messages)
    feedback = response.content
    
    print(f"👷🏽‍♂️ [QA Agent] Feedback: {feedback}\n")

    if "Pass" in feedback or retry_counter >= MAX_RETRIES:
        if retry_counter >= MAX_RETRIES:
            print("⚠️ [QA Agent] Max retries reached. Proceeding with the current result.")
        retry_counter = 0  # Reset for the next user query
        return "Pass"
    
    state['feedback'] = feedback
    return "Fail"


def result2nl(state: State, llm: ChatOCIGenAI) -> dict:
    """
    Node that converts the final JSON result into a natural language response.

    Args:
        state: The current state of the workflow.
        llm: The language model instance.

    Returns:
        A dictionary updating the state with the final natural language answer.
    """
    print("🙆‍♂️ [Formatter Agent] Converting JSON result to natural language...")
    prompt_template = PromptTemplate(
        input_variables=["question", "sql", "result"],
        template=RESULT_TO_NL_TEMPLATE,
        template_format="jinja2"
    )
    prompt_string = prompt_template.format(
        question=state['questionlatest'],
        sql=state['sqllatest'],
        result=state['resultlatest']
    )
    
    messages = [HumanMessage(content=prompt_string)]
    response = llm.invoke(messages)
    nl_text = response.content

    print(f"🙆‍♂️ [Formatter Agent] Final Answer: {nl_text}\n")
    return {"nl": nl_text}


# --- WORKFLOW ASSEMBLY AND EXECUTION ---

def create_langgraph_workflow(llm: ChatOCIGenAI, connection: oracledb.Connection):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(State)

    # Add nodes to the graph
    workflow.add_node("selectai", lambda state: selectai(state, connection))
    workflow.add_node("improve_question", lambda state: improve_question(state, llm))
    workflow.add_node("result2nl", lambda state: result2nl(state, llm))

    # Define the graph's control flow
    workflow.add_edge(START, "selectai")
    workflow.add_conditional_edges(
        "selectai",
        lambda state: check_selectai_output(state, llm),
        {"Pass": "result2nl", "Fail": "improve_question"}
    )
    workflow.add_edge("improve_question", "selectai")
    workflow.add_edge("result2nl", END)

    return workflow.compile()


def main():
    """
    Main function to set up services, build the workflow, and launch the UI.
    """
    llm = setup_llm()
    db_connection = setup_database_connection()
    
    app_workflow = create_langgraph_workflow(llm, db_connection)

    def respond(question, _history):
        """
        Invokes the LangGraph chain for a given question.
        
        Args:
            question: The user's question from the chat interface.
            _history: The chat history (managed by Gradio, unused here).

        Returns:
            The final natural language response from the workflow.
        """
        initial_state = {
            "questionlatest": question,
            "sqllatest": "",
            "resultlatest": "",
            "questionhistory": [],
            "sqlhistory": [],
            "resulthistory": [],
            "feedback": "",
            "nl": ""
        }
        final_state = app_workflow.invoke(initial_state)
        return final_state['nl']

    # Launch the Gradio Chat Interface
    gr.ChatInterface(
        fn=respond,
        title="Advanced Data Query",
        description="Ask a question about your medical trial data. The system will analyze, refine, and query to find your answer.",
    ).launch()

    # Clean up the database connection when the app is closed
    db_connection.close()
    print("🔌 Database connection closed.")


if __name__ == "__main__":
    main()
