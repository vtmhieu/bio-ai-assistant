"""
LangChain ReAct agent for biomedical research assistance.

This demonstrates the AI agent architecture pattern mentioned
in the SciLifeLab JD. The agent decides which tool to use
based on the researcher's question.
"""

from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from agent.tools import classify_biomedical_text, search_pubmed, summarize_text

# --- Option A: Use Ollama with a local LLM (recommended for M4 Mac) ---
# Install: brew install ollama && ollama pull llama3.2:3b
# This runs entirely locally — no API key needed.
llm = Ollama(model="llama3.2:3b", temperature=0)

# --- Option B: Use OpenAI API (if you have a key) ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the tools the agent can use
tools = [classify_biomedical_text, search_pubmed, summarize_text]

# ReAct prompt — the agent reasons step-by-step
REACT_PROMPT = PromptTemplate.from_template(
    """You are a biomedical research assistant
at SciLifeLab. You help researchers with their questions using available tools.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question
Thought: think about what tool to use
Action: the tool name
Action Input: the input to the tool
Observation: the result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}"""
)

# Create the agent
agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows the agent's reasoning — great for demos
    max_iterations=5,
    handle_parsing_errors=True,
)


def run_agent(question: str) -> str:
    """Run the agent on a research question."""
    result = agent_executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    # Test queries that exercise different tools
    test_queries = [
        "Classify this sentence: We enrolled 500 patients in a randomized controlled trial.",
        "Find recent papers about CRISPR gene therapy for cancer.",
        "What role does this sentence play in a paper: Our results demonstrate a significant improvement in survival rates.",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = run_agent(q)
        print(f"\nAnswer: {answer}")
