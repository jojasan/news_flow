from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from news_flow.main import NewsFlow

# In-memory store for tasks; note that this will be lost if the server restarts.
task_store = {}

class Task(BaseModel):
    id: Optional[str] = None
    news_urls: Optional[List[str]] = []
    num_starting_pool_news: Optional[int] = 2
    num_max_news: Optional[int] = 1
    topic: Optional[str] = ''
    perspective: Optional[str] = ''
    tone: Optional[str] = ''

app = FastAPI()

@app.post("/happifynews")
async def happify_news(task: Task, background_tasks: BackgroundTasks):
    # Validate that at least one of news_urls or topic is provided.
    if not task.news_urls and not task.topic:
        raise HTTPException(status_code=400, detail="Either news_urls or topic must be provided.")
    
    # Use the provided task.id if available; otherwise generate one.
    task_id = task.id if task.id is not None else str(len(task_store) + 1)
    task.id = task_id  # ensure the task carries this id
    
    # Initialize the task status as "processing".
    task_store[task_id] = {"status": "processing", "result": None}
    
    # Schedule the background workflow.
    background_tasks.add_task(kickoff_workflow, task, task_id)
    
    # Return the task_id along with all the input values provided.
    return {
        "task_id": task_id,
        "status": "processing",
        "news_urls": task.news_urls,
        "num_starting_pool_news": task.num_starting_pool_news,
        "num_max_news": task.num_max_news,
        "topic": task.topic,
        "perspective": task.perspective,
        "tone": task.tone,
    }

def kickoff_workflow(task: Task, task_id: str):
    news_flow = NewsFlow()
    # Build the inputs dictionary using the task_id as the id.
    inputs = {
        'id': task_id,  
        'num_starting_pool_news': task.num_starting_pool_news,
        'num_max_news': task.num_max_news,
        'current_date': '2025-03-07',
    }
    # If news_urls are provided, include them; otherwise, use topic and related parameters.
    if task.news_urls and len(task.news_urls) > 0:
        inputs['news_urls'] = task.news_urls
    elif task.topic:
        inputs['topic'] = task.topic
        inputs['perspective'] = task.perspective
        inputs['tone'] = task.tone

    # Execute the workflow and update the in-memory store.
    # (In a real scenario, this may take 30-40 minutes to complete.)
    result = news_flow.kickoff(inputs=inputs)
    task_store[task_id] = {"status": "completed", "result": result}

@app.get("/happifynews/{task_id}")
def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    task_info = task_store[task_id]
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "result": task_info["result"]
    }
