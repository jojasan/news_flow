from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import NewsResearchPlan, Ideas, Datapoints, CounterArguments
# from news_flow.tools.Crawl4AI import Crawl4AITool
from crewai.llm import LLM
from crewai_tools import FirecrawlScrapeWebsiteTool
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Tools
firecrawl = FirecrawlScrapeWebsiteTool(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class PlanningCrew:
    """Planning Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def editorial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["editorial_analyst"],
            llm=LLM(
                model="openai/o3-mini", 
                num_retries=3,
                fallbacks=[
                    {
                        "model": "openai/gpt-4o"
                    },
                ],
            ),
            tools=[firecrawl],
            max_rpm=10,
            verbose=True
        )

    @task
    def elicit_key_ideas_task(self) -> Task:
        return Task(
            config=self.tasks_config["elicit_key_ideas_task"],
            output_pydantic=Ideas,
        )
    
    @task
    def elicit_datapoints_task(self) -> Task:
        return Task(
            config=self.tasks_config["elicit_datapoints_task"],
            output_pydantic=Datapoints,
        )
    
    @task
    def create_counterarguments_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_counterarguments_task"],
            output_pydantic=CounterArguments,
        )
    
    @task
    def consolidate_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["consolidate_analysis_task"],
            output_pydantic=NewsResearchPlan,
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
        )
