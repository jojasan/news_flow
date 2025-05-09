from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from news_flow.llm_configs import o4_mini_high_reasoning
from news_flow.tools import tavily_search, tavily_scrape

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class WriterCrew:
    """Writer Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks/f_writer_tasks.yaml"

    @agent
    def expert_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["expert_writer"],
            llm=o4_mini_high_reasoning(),
            verbose=True,
            max_iter=10,
            tools=[tavily_search, tavily_scrape],
        )
    
    @task
    def write_article_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_article_task"],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Writer Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
