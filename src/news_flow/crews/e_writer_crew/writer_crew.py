from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from news_flow.llm_configs import o3_mini_high_reasoning

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class WriterCrew:
    """Writer Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def expert_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["expert_writer"],
            llm=o3_mini_high_reasoning(),
            verbose=True
        )
    
    @task
    def write_article_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_article_task"],
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
