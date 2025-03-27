from typing import Type, Any
import asyncio
import nest_asyncio
# nest_asyncio.apply() # lets see if this fixes the issues
import time

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

class Crawl4AIToolInput(BaseModel):
    """Input schema for Crawl4AITool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class Crawl4AITool(BaseTool):
    name: str = "Read website content"
    description: str = (
        "A tool that can be used to read a website content."
    )
    args_schema: Type[BaseModel] = Crawl4AIToolInput
    website_url: str = None

    def __init__(
        self,
        website_url: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.website_url = website_url
        self.description = (f"A tool that can be used to read {website_url}'s content.")
        self._generate_description()

    # Keep _run synchronous as required
    def _run(self, **kwargs: Any) -> str:
        website_url = kwargs.get("website_url", self.website_url)

        # time.sleep(5) # doing this to see if it can wait to attach itself the the existing loop!

        try:
            loop = asyncio.get_running_loop()
            nest_asyncio.apply(loop) # this patches the loop created by CrewAI to allow nested asyncio.run
        except RuntimeError:  # No running loop
            print("No running loop")
            return asyncio.run(self._crawl_content(website_url))
        else:  # If loop is already running. This is required since crewAI kickoff executes a asyncio.run(run_flow())
            print("Attaching to existing async loop")
            return loop.run_until_complete(self._crawl_content(website_url))

    # Async helper method
    async def _crawl_content(self, website_url: str) -> str:
        browser_conf = BrowserConfig(headless=True, # or True for headless mode
                                     user_agent_mode="random",
                                     viewport_width=1920,
                                     viewport_height=1080,)  
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")),
            # page_timeout=4000, #7 secs
            wait_for="js:() => window.loaded === true",
            scan_full_page=True,
            simulate_user=True,
            exclude_social_media_links=True,
        )
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=website_url, config=run_conf)
            markdown_content = result.markdown.fit_markdown  
            return markdown_content
