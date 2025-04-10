from typing import Type, Literal, Optional, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from tavily import TavilyClient
import os

class TavilySearchToolInput(BaseModel):
    """Input schema for TavilySearchTool."""
    search_terms: str = Field(..., description="The search terms to use for the search.")
    time_range: Optional[Literal['day', 'week', 'month', 'year']] = Field(default='day', description="The time range to use for the search, restricted to 'day', 'week', 'month', or 'year'.")

class TavilySearchTool(BaseTool):
    name: str = "Search the web with Tavily"
    description: str = ("Search the web for the given search terms.")
    topic: Optional[Literal['news', 'finance', '']] = Field(default='', description="The topic of the search, restricted to 'news', 'finance', or empty.")
    include_images: Optional[bool] = Field(default=False, description="Whether to include images in the search.")
    include_raw_content: Optional[bool] = Field(default=False, description="Whether to include the raw content of the search.")
    _tavily_client: Optional[TavilyClient] = PrivateAttr(default=None)
    args_schema: Type[BaseModel] = TavilySearchToolInput

    def __init__(
        self,
        topic: Optional[Literal['news', 'finance', 'general']] = Field(default='general', description="The topic of the search, restricted to 'news', 'finance', or empty."),
        include_images: Optional[bool] = Field(default=False, description="Whether to include images in the search."),
        include_raw_content: Optional[bool] = Field(default=False, description="Whether to include the raw content of the search."),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topic = topic
        self.include_images = include_images
        self.include_raw_content = include_raw_content
        self._tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

    def _run(self, 
             search_terms: str = Field(..., description="The search terms to use for the search."), 
             time_range: Optional[Literal['day', 'week', 'month', 'year']] = Field(default='day', description="The time range to use for the search, restricted to 'day', 'week', 'month', or 'year'.") ) -> str:
        response = self._tavily_client.search(
            query = search_terms,
            time_range = time_range,
            topic = self.topic,
            include_images = self.include_images,
            include_raw_content = self.include_raw_content
        )
        return response

class TavilyScrapeToolInput(BaseModel):
    """Input schema for TavilyScrapeTool."""
    urls: List[str] = Field(..., description="List of URLs to scrape.")

class TavilyScrapeTool(BaseTool):
    name: str = "Scrape websites with Tavily"
    description: str = ("Scrape websites for the given URLs.")
    include_images: Optional[bool] = Field(default=False, description="Whether to include images in the scrape.")
    args_schema: Type[BaseModel] = TavilyScrapeToolInput
    _tavily_client: Optional[TavilyClient] = PrivateAttr(default=None)

    def __init__(
        self,
        include_images: Optional[bool] = Field(default=False, description="Whether to include images in the search."),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_images = include_images
        self._tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

    def _run(self, urls: List[str] = Field(..., description="List of URLs to scrape.") ) -> str:
        response = self._tavily_client.extract(
            urls=urls,
            include_images=self.include_images
        )
        return response