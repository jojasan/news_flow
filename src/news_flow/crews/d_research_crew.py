from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import SupportingEvidence
from crewai.llm import LLM
from news_flow.tools import serper_search, brave_search, firecrawl, tavily_scrape
from news_flow.llm_configs import (
    gemini_flash_with_gpt4_1_mini_fallback,
    gpt4_1_mini_with_gemini_flash_fallback,
    o4_mini_with_gemini_flash_fallback
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ResearchCrew:
    """Research Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks/d_research_tasks.yaml"

    @agent
    def web_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst"],
            llm=gemini_flash_with_gpt4_1_mini_fallback(),
            tools=[serper_search, firecrawl],
            max_rpm=6, # to not overwhelm the firecrawl API
            verbose=True
        )
    
    @agent
    def web_research_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst_2"],
            llm=gpt4_1_mini_with_gemini_flash_fallback(),
            tools=[brave_search, tavily_scrape],
            max_rpm=6, # to not overwhelm the tavily API
            verbose=True
        )
    
    @agent
    def research_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["research_lead"],
            llm=o4_mini_with_gemini_flash_fallback(),
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
