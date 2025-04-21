from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import NewsResearchPlan, Ideas, Datapoints, CounterArguments
from crewai.llm import LLM
from news_flow.llm_configs import o4_mini_with_gpt4_1_fallback


@CrewBase
class PlanningCrew:
    """Planning Crew"""
    agents_config = "config/c_planning_agents.yaml"
    tasks_config = "config/c_planning_tasks.yaml"

    @agent
    def editorial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["editorial_analyst"],
            llm=o4_mini_with_gpt4_1_fallback(),
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
