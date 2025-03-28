from typing import Union
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from news_flow.main import NewsFlow

class Task(BaseModel):
    news_urls: Optional[List[str]] = []
    num_starting_pool_news: Optional[int] = 2
    num_max_news: Optional[int] = 1
    topic: Optional[str] = ''
    perspective: Optional[str] = ''
    tone: Optional[str] = ''

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/happifynews")
async def happify_news(task: Task, background_tasks: BackgroundTasks):

    # 'topic': 'Articificial Intelligence business case ROI in Banks',
    # 'topic': 'Climate Change in Colombia',
    # 'topic': 'Economic outlook of Peru',
    background_tasks.add_task(kickoff_workflow, task.topic)

    return {
        #"url": task.news_urls,
        "task_id": 12345,
        "status": "started",
        "num_starting_pool_news": task.num_starting_pool_news,
        "num_max_news": task.num_max_news,
        "topic": task.topic,
    }

def kickoff_workflow(task: Task):
    news_flow = NewsFlow()
    news_flow.kickoff(inputs={
        'id': 'api_call', # use an id if you want to start from the latest checkpoint
        'num_starting_pool_news': task.num_starting_pool_news,
        'num_max_news': task.num_max_news,
        'topic': task.topic,
        'perspective': task.perspective,
        'tone': task.tone,
        'current_date': '2025-03-07',
        # 'start_from_method': 'counter_args', # use this parameter to start from a specific method (starts after this one)
    })

# 'id': 'new_test_3', # use an id if you want to start from the latest checkpoint
# 'start_from_method': 'counter_args', # use this parameter to start from a specific method (starts after this one)
# 'num_starting_pool_news': 2,
# 'num_max_news': 1,
# # 'topic': 'Articificial Intelligence business case ROI in Banks',
# # 'topic': 'Climate Change in Colombia',
# 'topic': 'Economic outlook of Peru',
# 'perspective': 'Positive, optimistic',
# 'tone': 'Scientific, informative',
# 'current_date': '2025-03-07'