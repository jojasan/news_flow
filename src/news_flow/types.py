from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List, Set, Tuple, Any

class NewsWithSources(BaseModel):
    news_title: str = Field(..., description="The full name of the news.")
    summary: str = Field(..., description="The summary of the news.")
    source_url: str = Field(..., description="The url source for the news.")
    content: str = Field(..., description="The original unmodified content of the news from the source.")

class NewsList(BaseModel):
    news_list: List[NewsWithSources] = Field(..., description="A list of news objects (typed as NewsWithSources)")

class KeyIdea(BaseModel):
    key_idea: str = Field(..., description="The key idea extracted from the news.")
    rationale: str = Field(..., description="The rationale for why this is a key idea that supports the main argument.")

class Ideas(BaseModel):
    ideas: List[KeyIdea] = Field(..., description="A list of key ideas extracted from the news.")

class Datapoint(BaseModel):
    datapoint: str = Field(..., description="The quantitative datapoint extracted from the news.")
    rationale: str = Field(..., description="The rationale for why this is a key datapoint that supports the main argument.")

class Datapoints(BaseModel):
    datapoints: List[Datapoint] = Field(..., description="A list of key datapoints extracted from the news.")

class CounterArgument(BaseModel):
    counter_argument: str = Field(..., description="The counter argument extracted from the news.")
    rationale: str = Field(..., description="The rationale for why this is a counter argument that opposes the main argument.")

class CounterArguments(BaseModel):
    counter_arguments: List[CounterArgument] = Field(..., description="A list of counter arguments extracted from the news.")

class NewsResearchPlan(BaseModel):
    news_title: str = Field(..., description="The full name of the news.")
    source_url: str = Field(..., description="The url source for the news.")
    key_ideas: Ideas = Field(..., description="The key ideas extracted from the news.")
    key_datapoints: Datapoints = Field(..., description="The key datapoints extracted from the news.")
    counter_arguments: CounterArguments = Field(..., description="The counter arguments extracted from the news.")

class EvidenceItem(BaseModel):
    idea: str = Field(..., description="The key idea that is supported by the evidence.")
    evidence: str = Field(..., description="The evidence that supports the key idea.")
    source_url: str = Field(..., description="The url source for the evidence.") 
    rationale: str = Field(..., description="The rationale for why this evidence supports the key idea.")
    summary: str = Field(..., description="A summary of the evidence article content")

class SupportingEvidence(BaseModel):
    news_title: str = Field(..., description="The full name of the originally provided news article.")
    source_url: str = Field(..., description="The url source for the news.")
    idea: str = Field(..., description="The key idea that is supported by the evidence.")
    rationale: str = Field(..., description="The rationale for why this idea supports the main argument of the article")
    evidence: List[EvidenceItem] = Field(..., description="A list of evidence that supports the key idea. List items are of type 'EvidenceItem'")

class CounterArgSource(BaseModel):
    counter_argument: str = Field(..., description="The counter argument that opposes the main argument.")
    rationale: str = Field(..., description="The rationale for why this is a counter argument that opposes the main argument.")
    source_url: str = Field(..., description="The url source for the counter argument.")
    summary: str = Field(..., description="A summary of the counter argument article content")

class CounterArgumentSources(BaseModel):
    news_title: str = Field(..., description="The full name of the originally provided news article.")
    counter_argument: CounterArgument = Field(..., description="The provided counter argument extracted from the news.")
    supporting_sources: List[CounterArgSource] = Field(..., description="A list of sources that provide evidence for the counter argument. List items are of type 'CounterArgSource'")

# New consolidated models

class ConsolidatedNewsItem(BaseModel):
    news_title: str = Field(..., description="The unique news title")
    summary: Optional[str] = Field(None, description="Summary of the news")
    source_url: Optional[str] = Field(None, description="Source URL of the news")
    content: Optional[str] = Field(..., description="The original unmodified content of the news from the source.")
    supporting_evidence: List[SupportingEvidence] = Field(
        default_factory=list,
        description="List of SupportingEvidence objects corresponding to the news_title"
    )
    datapoints: List[Datapoints] = Field(
        default_factory=list,
        description="List of Datapoints objects corresponding to the news_title"
    )
    counter_argument_sources: List[CounterArgumentSources] = Field(
        default_factory=list,
        description="List of CounterArgumentSources objects corresponding to the news_title"
    )

class ConsolidatedNews(BaseModel):
    news_list: List[ConsolidatedNewsItem] = Field(
        ...,
        description="Consolidated list of news data grouped by news_title"
    )