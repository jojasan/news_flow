from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import CritiqueList
from crewai.llm import LLM
from news_flow.llm_configs import o4_mini_with_gpt4_1_fallback


@CrewBase
class CritiqueCrew:
    """Critique Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def editorial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["editorial_analyst"],
            llm=o4_mini_with_gpt4_1_fallback(),
            max_rpm=10,
            verbose=True
        )

    @task
    def critique_task(self) -> Task:
        return Task(
            config=self.tasks_config["critique_task"],
            output_pydantic=CritiqueList,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Critique Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
