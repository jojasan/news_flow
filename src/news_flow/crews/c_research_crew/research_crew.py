from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, YoutubeChannelSearchTool, YoutubeVideoSearchTool, BraveSearchTool
# from news_flow.tools.Crawl4AI import Crawl4AITool
from news_flow.types import SupportingEvidence
from crewai.llm import LLM
from crewai_tools import FirecrawlScrapeWebsiteTool
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Tools
serper_search = SerperDevTool()
brave_search = BraveSearchTool()
firecrawl = FirecrawlScrapeWebsiteTool(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class ResearchCrew:
    """Research Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def web_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst"],
            llm=LLM(
                model="openrouter/google/gemini-2.0-flash-001", 
                base_url="https://openrouter.ai/api/v1",
                num_retries=3,
                fallbacks=[
                    {
                        "model": "openai/gpt-4o-mini"
                    },
                    {
                        "model": "groq/llama-3.3-70b-versatile",
                    }
                ],
            ),
            tools=[serper_search, firecrawl],
            max_rpm=10,
            verbose=True
        )
    
    @agent
    def web_research_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst_2"],
            llm=LLM(
                model="openai/gpt-4o-mini", 
                num_retries=3,
                fallbacks=[
                    {
                        "model": "openrouter/google/gemini-2.0-flash-001",
                        "base_url": "https://openrouter.ai/api/v1",
                    },
                    {
                        "model": "groq/llama-3.3-70b-versatile"
                    },
                ],
            ),
            tools=[brave_search, firecrawl],
            max_rpm=10,
            verbose=True
        )
    
    @agent
    def research_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["research_lead"],
            llm=LLM(
                model="openai/o3-mini", 
                num_retries=3,
                fallbacks=[
                    {
                        "model": "openrouter/google/gemini-2.0-flash-001",
                        "base_url": "https://openrouter.ai/api/v1",
                    },
                    {
                        "model": "openai/gpt-4o"
                    },
                ],
            ),
            # allow_delegation=True,
            tools=[serper_search, firecrawl],
            max_iter=5,
            verbose=True
        )

    @task
    def find_supporting_resources_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_supporting_resources_task"],
            async_execution=True,
            # output_pydantic=SupportingEvidence
        )
    
    @task
    def find_supporting_resources_task_2(self) -> Task:
        return Task(
            config=self.tasks_config["find_supporting_resources_task_2"],
            async_execution=True,
            #output_pydantic=SupportingEvidence
        )
    
    @task
    def create_final_report(self) -> Task:
        return Task(
            config=self.tasks_config["create_final_report"],
            context=[self.find_supporting_resources_task(),self.find_supporting_resources_task_2()],
            output_pydantic=SupportingEvidence
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Planning Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # output_log_file="research_crew_logs.txt"
        )
