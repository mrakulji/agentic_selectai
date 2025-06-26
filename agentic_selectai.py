from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import oracledb
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from langchain_core.messages import HumanMessage

load_dotenv()

nl = '\n'

DB_CONFIG = {
    "user": "admin",
    "password": "<db password>",
    "dsn": "TSTAI_HIGH",
    "config_dir": "<config dir>",
    "wallet_location": "<wallet location>",
    "wallet_password": "<wallet password>",
    "thick_mode": "False"
}

connection = oracledb.connect(
    config_dir=DB_CONFIG["config_dir"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    dsn=DB_CONFIG["dsn"],
    wallet_location=DB_CONFIG["wallet_location"],
    wallet_password=DB_CONFIG["wallet_password"]
)

#Iteration counter
c = 0

medical_term_dict = """
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

improvequestionstr = """You are a highly skilled medical language simplification expert. Your task is to take complex, medically worded clinical questions and rephrase them into simpler, more direct natural language questions, **using a provided medical terminology translation dictionary to replace medical terms with their standard medical abbreviations.**

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

```
"What is the average systolic blood pressure for subjects at baseline visit who are in the Cardiovex treatment arm?"
```

**Example Expected Output:**

```
"What is the average SYSBP for patients at their first visit who are receiving Cardiovex treatment?"
```

**Now, please reword the following medically worded clinical question into a simpler, more direct natural language question, remembering to avoid any table or column names and maintain the original intent:**

Question: {question}
Feedback: {feedback}
Question History: {questionhistory}

Medical Terminology Translation Dictionary:
{medical_term_dictionary}
"""


gatepromptstr = """You are a highly skilled SQL Quality Assurance (QA) expert. Your task is to evaluate if a provided SQL query accurately answers a user's natural language question, based on the question and the **actual JSON formatted result** of executing that query.

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
Simulated SQL Result: {{ result }}"""


result2nlstr = """You are an expert in generating clear and concise natural language answers from structured data. Your task is to take a user's original question, the SQL query that was intended to answer it, and the JSON formatted result of executing that SQL query, and produce a human-readable answer to the user's question.

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
import gradio as gr

improve_question_prompt = PromptTemplate(
    input_variables=["question", "questionhistory", "feedback", "medical_term_dictionary"],
    template=improvequestionstr
)

check_selectai_output_prompt = PromptTemplate(
    input_variables=["question", "result"],
    template=gatepromptstr,
    template_format="jinja2"
)

result2nl_prompt = PromptTemplate(
    input_variables=["question", "sql", "result"],
    template=result2nlstr,
    template_format="jinja2"
)

llm = ChatOCIGenAI(
    model_id="<LLM model id>",
    service_endpoint="<OCI GenAI service endpoint>",
    compartment_id="<compartment id>",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)

# Graph state
class State(TypedDict):
    questionlatest: str
    sqllatest: str
    resultlatest: str
    questionhistory: list
    sqlhistory: list
    resulthistory: list
    feedback: str
    nl: str


# Nodes
def selectai(state: State):
    """Create a SQL Query and execute it"""

    query = """SELECT DBMS_CLOUD_AI.GENERATE(
                prompt       => :prompt,
                profile_name => 'OCI_GENAI_PROFILE',
                action       => :action)
            FROM dual"""
    
    print(f"💿 ### NEW QUESTION ###: {state['questionlatest']}{nl}")

    ##For demo purposes this step explains the SQL generation process   
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, {'prompt': state['questionlatest'], 'action': 'explainsql'})
            result = cursor.fetchone()
            if result and isinstance(result[0], oracledb.LOB):
                text_result = result[0].read()
                print(f"💿 ### SelectAI Explain Query ###: {text_result}{nl}")
                state['sqllatest'] = text_result[:3000]
    except Exception as e:
        return {"sqllatest": "NONE", "resultlatest": "NONE"}
    
    ##SQL execution
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, {'prompt': state['questionlatest'], 'action': 'runsql'})
            result = cursor.fetchone()
            if result and isinstance(result[0], oracledb.LOB):
                text_result = result[0].read()
                print(f"💿 ### SelectAI Result ###: {text_result}{nl}")
                state['resultlatest'] = text_result[:3000]
        #print(f" ### Select AI Question ###: {state['questionlatest']}{nl}, ### SQL ###: {state['sqllatest']}{nl}, ### Result ###: {state['resultlatest']}{nl}")
        return {"sqllatest": state['sqllatest'], "resultlatest": state['resultlatest']}
    except Exception as e:
        # return {f"Error: {e}"}
        print(f"💿 SelectAI: Result: NONE{nl}")
        return {"sqllatest": state['sqllatest'], "resultlatest": "NONE"}

def improve_question(state: State):
    """LLM call to improve the question"""
    print("Question History: ", state['questionhistory'])
    prompt_string = improve_question_prompt.format(question=state['questionlatest'], questionhistory=state['questionhistory'], feedback=state['feedback'], medical_term_dictionary=medical_term_dict)
    messages = [HumanMessage(content=prompt_string)] # Explicitly create HumanMessage

    #print(f"👨🏼‍🔧 Improve AI - Prompt to LLM: {prompt_string}{nl}")

    question_response = llm.invoke(messages)
    question = question_response.content

    state['questionhistory'].append(state['questionlatest'])
    state['questionlatest'] = question

    print(f"👨🏼‍🔧 ### Improve Question ###: {state['questionlatest']}{nl}")
      
    return {"questionlatest": question, "questionhistory": state['questionhistory']}

# Conditional edge function to check the Select AI output
def check_selectai_output(state: State):
    """Gate function to check if the correct sql and result has been produced"""

    global c
    c = c + 1
    prompt_string = check_selectai_output_prompt.format(question=state['questionlatest'], result=state['resultlatest'])
    messages = [HumanMessage(content=prompt_string)] # Explicitly create HumanMessage

    #print(f"👷🏽‍♂️ Check AI - Prompt to LLM: {prompt_string}{nl}")

    result = llm.invoke(messages)

    print(f"👷🏽‍♂️ ### Check AI - Feedback ###: {result.content}{nl}")
  
    # Stop the query improvement at 5 iterations 
    if "Pass" in result.content or c > 5:
        c = 0
        return "Pass"
    state['feedback'] = result.content
    return "Fail"

def result2nl(state: State):
    """LLM call to convert the answer to natural language"""

    prompt_string = result2nl_prompt.format(question=state['questionlatest'], sql=state['sqllatest'], result=state['resultlatest'])
    messages = [HumanMessage(content=prompt_string)] # Explicitly create HumanMessage

    #print(f"🙆‍♂️ JSON2NL AI - Prompt to LLM: {prompt_string}{nl}")

    nl_response = llm.invoke(messages)
    nl_text = nl_response.content

    print(f"🙆‍♂️ ### SelectAI to Natural Language Response ###: {nl_text}{nl}")

    return {"nl": nl_text}


from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Build workflow
workflow = StateGraph(State)

workflow.add_node("selectai", selectai)
workflow.add_node("improve_question", improve_question)
workflow.add_node("result2nl", result2nl)

workflow.add_edge(START, "selectai")

workflow.add_conditional_edges(
    "selectai", check_selectai_output, {"Pass": "result2nl", "Fail": "improve_question"}
)
workflow.add_edge("improve_question", "selectai")
workflow.add_edge("result2nl", END)

chain = workflow.compile()

def respond(question, history):
    state = chain.invoke({"questionlatest": question, "sqllatest": "", "resultlatest": "", "questionhistory": [], "sqlhistory": [], "resulthistory": [], "feedback": ""})
    return state['nl']

# Chat interface
gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Advanced Data Query",
    description="Ask a question about your medical trial data, and through a process of analysis and refinement, get a natural language answer.",
).launch()
