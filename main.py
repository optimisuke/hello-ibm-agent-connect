import os
import queue
import threading
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import StreamingResponse
import uvicorn
import json
import time
import uuid
import asyncio
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent
)
from crewai.utilities.events.base_event_listener import BaseEventListener

# Initialize FastAPI app
app = FastAPI(title="CrewAI Agent Connect Example")

# Define message schema


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


# Create CrewAI agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate and relevant information",
    backstory="You are an expert researcher with a talent for finding information.",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Create engaging and informative content",
    backstory="You are a skilled writer who can explain complex topics clearly.",
    verbose=True
)

analyst = Agent(
    role="Analyst",
    goal="Analyze information and extract insights",
    backstory="You are an analytical thinker who can identify patterns and insights.",
    verbose=True
)

# Create a function to process requests with CrewAI


def process_with_crew(query):
    # Create tasks
    research_task = Task(
        description=f"Research information about: {query}",
        agent=researcher,
        expected_output='A detailed research report.'
    )

    analysis_task = Task(
        description="Analyze the research findings and extract key insights",
        agent=analyst,
        dependencies=[research_task],
        expected_output='A summary of key findings.'
    )

    writing_task = Task(
        description="Create a comprehensive response based on the research and analysis",
        agent=writer,
        dependencies=[analysis_task],
        expected_output='A concise answer to the query with sources cited and insights discovered.'
    )

    # Create the crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential
    )

    # Run the crew
    result = crew.kickoff()
    return result

# Agent discovery endpoint


@app.get("/v1/agents")
async def discover_agents():
    return {
        "agents": [
            {
                "name": "CrewAI Research Team",
                "description": "A team of specialized agents that collaborate to research, analyze, and present information",
                "provider": {
                    "organization": "Your Organization",
                    "url": "https://your-organization.com"
                },
                "version": "1.0.0",
                "documentation_url": "https://docs.example.com/crewai-agent",
                "capabilities": {
                    "streaming": True
                }
            }
        ]
    }

# Chat completion endpoint


@app.post("/v1/chat")
async def chat_completion(request: ChatRequest, x_thread_id: str = Header(None)):
    thread_id = x_thread_id or str(uuid.uuid4())

    # Extract the user query from the messages
    user_messages = [msg for msg in request.messages if msg["role"] == "user"]
    if not user_messages:
        return {"error": "No user message found"}

    query = user_messages[-1]["content"]

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_crew_response(query, thread_id),
            media_type="text/event-stream"
        )
    else:
        # Process with CrewAI
        result = process_with_crew(query)

        # Format the response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return response


# Create a custom event listener for streaming CrewAI events
class CrewAIStreamingListener(BaseEventListener):
    def __init__(self, thread_id):
        super().__init__()
        self.thread_id = thread_id
        # Use a thread-safe queue for communication between threads
        self.thread_queue = queue.Queue()
        # Track events to prevent duplicates
        self.processed_events = set()
        print(
            f"CrewAIStreamingListener initialized with thread_id: {thread_id}")

    def setup_listeners(self, crewai_event_bus):
        print("Setting up event listeners")

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_kickoff_started(source, event):
            # Generate a unique event ID
            event_id = f"kickoff-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: CrewKickoffStartedEvent {event_id}")
            thinking_step = {
                "id": f"step-{uuid.uuid4()}",
                "object": "thread.run.step.delta",
                "thread_id": self.thread_id,
                "model": "crewai-team",
                "created": int(time.time()),
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "step_details": {
                                "type": "thinking",
                                "content": f"Assembling a team of agents to work on your request..."
                            }
                        }
                    }
                ]
            }
            # Use thread-safe queue.put() instead of asyncio.Queue
            print(f"Putting CrewKickoffStartedEvent {event_id} in queue")
            self.thread_queue.put(
                (f"event: thread.run.step.delta\n", f"data: {json.dumps(thinking_step)}\n\n"))

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event):
            # Generate a unique event ID
            event_id = f"agent-start-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: AgentExecutionStartedEvent {event_id}")
            agent_step = {
                "id": f"step-{uuid.uuid4()}",
                "object": "thread.run.step.delta",
                "thread_id": self.thread_id,
                "model": "crewai-team",
                "created": int(time.time()),
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "step_details": {
                                "type": "thinking",
                                "content": f"{event.agent.role} is working on: {event.task.description}"
                            }
                        }
                    }
                ]
            }
            self.thread_queue.put(
                (f"event: thread.run.step.delta\n", f"data: {json.dumps(agent_step)}\n\n"))

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event):
            # Generate a unique event ID
            event_id = f"tool-start-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: ToolUsageStartedEvent {event_id}")
            tool_step = {
                "id": f"step-{uuid.uuid4()}",
                "object": "thread.run.step.delta",
                "thread_id": self.thread_id,
                "model": "crewai-team",
                "created": int(time.time()),
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "step_details": {
                                "type": "tool_calls",
                                "tool_calls": [
                                    {
                                        "id": f"call-{uuid.uuid4()}",
                                        "name": event.tool_name,
                                        "args": event.tool_input
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
            self.thread_queue.put(
                (f"event: thread.run.step.delta\n", f"data: {json.dumps(tool_step)}\n\n"))

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event):
            # Generate a unique event ID
            event_id = f"tool-finish-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: ToolUsageFinishedEvent {event_id}")
            tool_response_step = {
                "id": f"step-{uuid.uuid4()}",
                "object": "thread.run.step.delta",
                "thread_id": self.thread_id,
                "model": "crewai-team",
                "created": int(time.time()),
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "step_details": {
                                "type": "tool_response",
                                "content": str(event.output),
                                "name": event.tool_name,
                                "tool_call_id": f"call-{uuid.uuid4()}"
                            }
                        }
                    }
                ]
            }
            self.thread_queue.put(
                (f"event: thread.run.step.delta\n", f"data: {json.dumps(tool_response_step)}\n\n"))

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event):
            # Generate a unique event ID
            event_id = f"agent-complete-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: AgentExecutionCompletedEvent {event_id}")
            # Stream the agent's output as a message
            message_chunks = split_into_chunks(event.output)

            for i, chunk in enumerate(message_chunks):
                message_delta = {
                    "id": f"msg-{uuid.uuid4()}",
                    "object": "thread.message.delta",
                    "thread_id": self.thread_id,
                    "model": "crewai-team",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": chunk
                            }
                        }
                    ]
                }
                self.thread_queue.put(
                    (f"event: thread.message.delta\n", f"data: {json.dumps(message_delta)}\n\n"))

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_completed(source, event):
            # Generate a unique event ID
            event_id = f"kickoff-complete-{id(event)}"

            # Check if we've already processed this event
            if event_id in self.processed_events:
                print(f"Skipping duplicate event: {event_id}")
                return

            # Mark this event as processed
            self.processed_events.add(event_id)

            print(f"Event: CrewKickoffCompletedEvent {event_id}")
            # Signal that the streaming is complete
            self.thread_queue.put(None)

# Stream response function for CrewAI


async def stream_crew_response(query, thread_id):
    # Create the event listener
    listener = CrewAIStreamingListener(thread_id)

    # Start the crew execution in a separate thread
    thread = threading.Thread(
        target=process_with_crew_thread,
        args=(query, listener)
    )
    thread.daemon = True
    thread.start()

    # Create an asyncio queue for the FastAPI streaming response
    async_queue = asyncio.Queue()

    # Start a background task to transfer items from thread queue to async queue
    asyncio.create_task(transfer_queue_items(
        listener.thread_queue, async_queue))

    # Stream events from the async queue
    while True:
        event_data = await async_queue.get()
        if event_data is None:
            break

        event_type, event_content = event_data
        yield event_type
        yield event_content

# Function to transfer items from thread queue to async queue


async def transfer_queue_items(thread_queue, async_queue):
    print("Starting queue transfer task")
    while True:
        try:
            # Use blocking get with timeout to be more responsive
            # but not consume too much CPU
            item = thread_queue.get(block=True, timeout=0.1)
            print(
                f"Received item from thread queue: {item[:50] if item is not None else None}")

            # If we got None, it means we're done
            if item is None:
                print("Received None, signaling end of streaming")
                await async_queue.put(None)
                break

            # Otherwise, put the item in the async queue
            await async_queue.put(item)
            print(f"Put item in async queue")

        except queue.Empty:
            # If timeout occurs, just continue the loop
            await asyncio.sleep(0.01)  # Short sleep to prevent CPU spinning

# Function to run CrewAI in a separate thread


def process_with_crew_thread(query, listener):
    try:
        print("Starting CrewAI execution in thread")

        # Import the event bus to register the listener
        from crewai.utilities.events import crewai_event_bus

        # Register the listener with the event bus
        print("Registering event listener")
        listener.setup_listeners(crewai_event_bus)
        print("Event listener registered")

        # Create tasks
        print("Creating tasks")
        research_task = Task(
            description=f"Research information about: {query}",
            agent=researcher,
            expected_output='A detailed research report.'
        )

        analysis_task = Task(
            description="Analyze the research findings and extract key insights",
            agent=analyst,
            dependencies=[research_task],
            expected_output='A summary of key findings.'
        )

        writing_task = Task(
            description="Create a comprehensive response based on the research and analysis",
            agent=writer,
            dependencies=[analysis_task],
            expected_output='A concise answer to the query with sources cited and insights discovered.'
        )

        # Create the crew
        print("Creating crew")
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential
        )

        # Run the crew directly in this thread
        print("Starting crew.kickoff()")
        result = crew.kickoff()
        print("Crew execution completed")

        return result
    except Exception as e:
        # In case of error, put a None in the queue to signal the end of streaming
        print(f"Error in crew execution: {str(e)}")
        listener.thread_queue.put(None)

# Helper function to split text into chunks


def split_into_chunks(text, chunk_size=10):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
