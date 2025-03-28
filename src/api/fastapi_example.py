from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from typing import List, Optional
import logging
import asyncio
from functools import lru_cache
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="CrewAI API Service")

class CrewRequest(BaseModel):
    objective: str
    context: Optional[str] = None
    max_iterations: Optional[int] = 3
    async_execution: Optional[bool] = True

class CrewResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None

# In-memory store for task results
task_results = {}

@lru_cache()
def get_agents():
    """Cache and return agent instances to reduce initialization overhead"""
    researcher = Agent(
        role='Researcher',
        goal='Find and analyze relevant information',
        backstory='Expert at gathering and analyzing information',
        allow_delegation=False
    )
    
    writer = Agent(
        role='Writer',
        goal='Create clear and concise content',
        backstory='Expert at creating engaging content',
        allow_delegation=False
    )
    
    return [researcher, writer]

async def process_crew_task(task_id: str, crew_request: CrewRequest):
    """Execute crew task asynchronously"""
    try:
        # Initialize agents
        agents = get_agents()
        
        # Create tasks
        research_task = Task(
            description=f"Research: {crew_request.objective}",
            agent=agents[0]
        )
        
        writing_task = Task(
            description=f"Write content about: {crew_request.objective}",
            agent=agents[1]
        )
        
        # Initialize crew
        crew = Crew(
            agents=agents,
            tasks=[research_task, writing_task],
            max_iterations=crew_request.max_iterations
        )
        
        # Execute crew tasks
        result = await crew.kickoff()
        
        # Store result
        task_results[task_id] = {
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        task_results[task_id] = {
            "status": "failed",
            "result": str(e)
        }

@app.post("/crew/tasks", response_model=CrewResponse)
async def create_crew_task(
    crew_request: CrewRequest,
    background_tasks: BackgroundTasks
):
    """Create and initiate a new crew task"""
    task_id = str(len(task_results) + 1)  # Simple ID generation
    
    if crew_request.async_execution:
        # Initialize task status
        task_results[task_id] = {
            "status": "processing",
            "result": None
        }
        
        # Schedule task execution
        background_tasks.add_task(
            process_crew_task,
            task_id,
            crew_request
        )
        
        return CrewResponse(
            task_id=task_id,
            status="processing"
        )
    else:
        # Synchronous execution
        await process_crew_task(task_id, crew_request)
        result = task_results[task_id]
        
        return CrewResponse(
            task_id=task_id,
            status=result["status"],
            result=result["result"]
        )

@app.get("/crew/tasks/{task_id}", response_model=CrewResponse)
async def get_task_status(task_id: str):
    """Get status and result of a specific task"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_results[task_id]
    return CrewResponse(
        task_id=task_id,
        status=result["status"],
        result=result["result"]
    )

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize any resources (e.g., connection pools)
    logger.info("Starting up CrewAI API Service")
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down CrewAI API Service")