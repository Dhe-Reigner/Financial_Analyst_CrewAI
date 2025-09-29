import re
import json
import os
import yfinance as yf
from pydantic import BaseModel,Field
from crewai import Agent,Task,Crew,Process,LLM
from crewai_tools import FileReadTool,CodeInterpreterTool
from dotenv import load_dotenv

load_dotenv()

class QueryAnalysisOutput(BaseModel):
    """Structured output for the query analysis task"""
    symbols:list[str] = Field(..., description="List of stock ticker symbols(eg.['TSLA','AAPL'])")
    timeframe:str = Field(...,description="Time period(eg,'1d','1mo','1y)")
    action:str = Field(...,description="Action to be performed (eg.'fetch','plot')")

llm = LLM(
    model="ollama/phi:2.7b",
    base_url="http://localhost:11434",
    temperature=0.7
)
# 1) Query Parser Agent
query_parser_agent = Agent(
    role="Stock Data Analyst",
    goal="EXtract stock details and fetch required data from this user query:{query}.",
    backstory="You are a financial analyst specializing in stock market data retrieval.",
    llm=llm,
    verbose=True,
    memory=True,
)
query_parsing_task=Task(
    description="Analyse the user query and extract stock details.",
    expected_output="A dictionary with keys:'symbol','timeframe','action'.",
    output_pydantic=QueryAnalysisOutput,
    agent=query_parser_agent
)
# 2) Code writer agent
code_writer_agent = Agent(
    role="Senior Python Developer",
    goal="Write Python code to visualize stock data.",
    backstory="""
     You are a Senior Python developer specializing in stock market data visualization.
     You are also a Pandas, Matplotlib and yfinance library expert.
     You are skilled at writing production-ready Python code
     """,
    llm=llm,
    verbose=True
)
code_writer_task = Task(
    description="""
     Write Python code to visualize stock data based on the inputs from the stock analyst
     where you would find stock symbol, timeframe and action.
    """,
    expected_output="A clean and executable Python script file(.py) for stock visualization.",
    agent=code_writer_agent
)

# 3).Code interpreter agent(uses code interpreter tool from crewai)
code_interpreter_tool = CodeInterpreterTool(use_docker=False)

code_execution_agent = Agent(
    role="Senior Code EXecution Expert",
    goal="Review and execute the generated Python code by code writer agent to visualize stock data and fix any errors encountered. It can delegate tasks to code writer agent if needed",
    backstory="You are a code execution expert. You are skilled at executing Python code.",
    tools=[code_interpreter_tool],
    allow_delegation=True,
    llm=llm,
    verbose=True
)
code_execution_task = Task(
    description="""Review and execute the generated Python code by code writer agent to visualize stock data and fix any errors encountered.""",
    expected_output="A clean, working and executable Python script file (.py) for stock visulization.",
    agent=code_execution_agent,
)

# Create the crew
crew = Crew(
    agents=[query_parser_agent,code_writer_agent,code_execution_agent],
    tasks=[query_parsing_task,code_writer_task,code_execution_task],
    process=Process.sequential
)

def run_fianncial_analysis(query):
    result = crew.kickoff(inputs={'query':query})
    return result.raw

if __name__=="__main__":
    result = crew.kickoff(inputs={"query":"Plot YTD stock gain of Tesla"})
    print(result.raw)