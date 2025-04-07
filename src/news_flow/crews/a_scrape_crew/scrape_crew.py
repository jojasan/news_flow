from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from news_flow.types import NewsList
from crewai.llm import LLM
from crewai_tools import FirecrawlScrapeWebsiteTool
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Tools
firecrawl = FirecrawlScrapeWebsiteTool(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)

@CrewBase
class ScrapeCrew:
    """Scrape Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def news_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["news_scraper"],
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
            verbose=True
        )
    
    @task
    def scrape_news_task(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_news_task"],
            output_pydantic=NewsList,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Scrape Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
