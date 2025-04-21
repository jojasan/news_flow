from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import NewsList
from crewai.llm import LLM
from news_flow.tools import serper_search, firecrawl, tavily_search, tavily_scrape
from news_flow.llm_configs import (
    o4_mini_with_gemini_flash_fallback, 
    gemini_flash_with_gpt4_1_mini_fallback
)

@CrewBase
class DiscoverCrew:
    """Discover Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_expert"],
            llm=o4_mini_with_gemini_flash_fallback(),
            max_iter=5,
            tools=[serper_search, firecrawl],
            verbose=True
        )

    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],
            llm=gemini_flash_with_gpt4_1_mini_fallback(),
            max_iter=5,
            tools=[tavily_search, tavily_scrape],
            verbose=True
        )

    @task
    def identify_news_outlets_task(self) -> Task:
        return Task(
            config=self.tasks_config["identify_news_outlets_task"],
            async_execution=True,
        )

    @task
    def general_web_search_task(self) -> Task:
        return Task(
            config=self.tasks_config["general_web_search_task"],
            async_execution=True
        )
    
    @task
    def search_specialized_sources_task(self) -> Task:
        return Task(
            config=self.tasks_config["search_specialized_sources_task"],
            context=[self.identify_news_outlets_task()]
        )
    
    @task
    def prioritize_news_task(self) -> Task:
        return Task(
            config=self.tasks_config["prioritize_news_task"],
            context=[self.search_specialized_sources_task(), self.general_web_search_task()], 
            output_pydantic=NewsList,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Discover Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # output_log_file="discover_crew_logs.txt"
        )
