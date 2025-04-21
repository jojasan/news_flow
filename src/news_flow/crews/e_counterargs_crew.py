from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import CounterArgumentSources
from crewai.llm import LLM
from news_flow.tools import serper_search, brave_search, firecrawl, tavily_scrape
from news_flow.llm_configs import (
    gemini_flash_with_gpt4_1_mini_fallback,
    gpt4_1_mini_with_gemini_flash_fallback,
    o4_mini_with_gemini_flash_fallback
)

@CrewBase
class CounterArgumentsCrew:
    """CounterArguments Crew"""
    agents_config = "config/e_counterargs_agents.yaml"
    tasks_config = "config/e_counterargs_tasks.yaml"

    @agent
    def web_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst"],
            llm=gemini_flash_with_gpt4_1_mini_fallback(),
            tools=[serper_search, firecrawl],
            max_rpm=4,
            verbose=True
        )
    
    @agent
    def web_research_analyst_2(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_analyst_2"],
            llm=gpt4_1_mini_with_gemini_flash_fallback(),
            tools=[brave_search, tavily_scrape],
            max_rpm=4,
            verbose=True
        )
    
    @agent
    def research_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["research_lead"],
            llm=o4_mini_with_gemini_flash_fallback(),
            tools=[serper_search, firecrawl],
            max_iter=5,
            verbose=True
        )
    
    @task
    def research_counterarguments_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_counterarguments_task"],
            async_execution=True,
            output_pydantic=CounterArgumentSources
        )
    
    @task
    def research_counterarguments_task_2(self) -> Task:
        return Task(
            config=self.tasks_config["research_counterarguments_task_2"],
            async_execution=True,
            output_pydantic=CounterArgumentSources
        )
    
    @task
    def consolidate_counterarg_resources_task(self) -> Task:
        return Task(
            config=self.tasks_config["consolidate_counterarg_resources_task"],
            context=[self.research_counterarguments_task(), self.research_counterarguments_task_2()],
            output_pydantic=CounterArgumentSources
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the CounterArgs Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # output_log_file="counterargs_crew_logs.txt"
        )
