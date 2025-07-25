"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import base64
import logging
import os
import re

from collections.abc import AsyncIterable
from io import BytesIO
from typing import Any
from uuid import uuid4

from PIL import Image
from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from crewai.tools import tool
from dotenv import load_dotenv
from google import genai
from google.genai import types
from in_memory_cache import InMemoryCache
from pydantic import BaseModel
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


load_dotenv()

logger = logging.getLogger(__name__)


class Imagedata(BaseModel):
    """Represents image data.

    Attributes:
      id: Unique identifier for the image.
      name: Name of the image.
      mime_type: MIME type of the image.
      bytes: Base64 encoded image data.
      error: Error message if there was an issue with the image.
    """

    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    bytes: str | None = None
    error: str | None = None


@tool('ImageGenerationTool')
def generate_image_tool(
    prompt: str, session_id: str, artifact_file_id: str = None
) -> str:
    """Image generation tool that generates images or modifies a given image based on a prompt."""
    if not prompt:
        raise ValueError('Prompt cannot be empty')

    client = genai.Client()
    cache = InMemoryCache()

    text_input = (
        prompt,
        'Ignore any input images if they do not match the request.',
    )

    ref_image = None
    logger.info(f'Session id {session_id}')
    print(f'Session id {session_id}')

    # TODO (rvelicheti) - Change convoluted memory handling logic to a better
    # version.
    # Get the image from the cache and send it back to the model.
    # Assuming the last version of the generated image is applicable.
    # Convert to PIL Image so the context sent to the LLM is not overloaded
    try:
        ref_image_data = None
        # image_id = session_cache[session_id][-1]
        session_image_data = cache.get(session_id)
        if artifact_file_id:
            try:
                ref_image_data = session_image_data[artifact_file_id]
                logger.info('Found reference image in prompt input')
            except Exception:
                ref_image_data = None
        if not ref_image_data:
            # Insertion order is maintained from python 3.7
            latest_image_key = list(session_image_data.keys())[-1]
            ref_image_data = session_image_data[latest_image_key]

        ref_bytes = base64.b64decode(ref_image_data.bytes)
        ref_image = Image.open(BytesIO(ref_bytes))
    except Exception:
        ref_image = None

    if ref_image:
        contents = [text_input, ref_image]
    else:
        contents = text_input

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            ),
        )
    except Exception as e:
        logger.error(f'Error generating image {e}')
        print(f'Exception {e}')
        return -999999999

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            try:
                print('Creating image data')
                data = Imagedata(
                    bytes=base64.b64encode(part.inline_data.data).decode(
                        'utf-8'
                    ),
                    mime_type=part.inline_data.mime_type,
                    name='generated_image.png',
                    id=uuid4().hex,
                )
                session_data = cache.get(session_id)
                if session_data is None:
                    # Session doesn't exist, create it with the new item
                    cache.set(session_id, {data.id: data})
                else:
                    # Session exists, update the existing dictionary directly
                    session_data[data.id] = data

                return data.id
            except Exception as e:
                logger.error(f'Error unpacking image {e}')
                print(f'Exception {e}')
    return -999999999


@tool('WebSearchTool')
def web_search_tool(query: str) -> str:
    """Search the web for the given query using Serper."""
    return SerperDevTool().run(query)

@tool('WebsiteScrapeTool')
def website_scrape_tool(url: str) -> str:
    """Scrape the content of the given website URL."""
    return ScrapeWebsiteTool().run(url)

class FinanceResearchAgent:
    """Agent that researches a topic and analyzes its financial aspects."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        if os.getenv('GOOGLE_GENAI_USE_VERTEXAI'):
            self.model = LLM(model='vertex_ai/gemini-2.0-flash')
        elif os.getenv('GOOGLE_API_KEY'):
            self.model = LLM(
                model='gemini/gemini-2.0-flash',
                api_key=os.getenv('GOOGLE_API_KEY'),
            )

        self.research_agent = Agent(
            role='Financial Research Analyst',
            goal=(
                "For any given topic, research and analyze the financial aspects, trends, or impacts. "
                "Use web search and website scraping tools to gather information, then summarize your findings "
                "with a focus on finance."
            ),
            backstory=(
                "You are an expert financial analyst. You use the latest web search and scraping tools to gather "
                "and synthesize information, always focusing on the financial implications, market trends, and "
                "economic impacts of any topic you are asked about."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            llm=self.model,
        )

        self.research_task = Task(
            description=(
                "Receive a user topic: '{user_topic}'.\n"
                "1. Search the web for the topic using the Web Search Tool.\n"
                "2. Scrape relevant websites using the Website Scrape Tool.\n"
                "3. Summarize the findings, focusing on the financial aspects, trends, or impacts.\n"
                "Return a concise financial analysis summary."
            ),
            expected_output='A concise summary of the financial aspects of the topic.',
            agent=self.research_agent,
        )

        self.crew = Crew(
            agents=[self.research_agent],
            tasks=[self.research_task],
            process=Process.sequential,
            verbose=False,
        )

    def extract_artifact_file_id(self, query):
        try:
            pattern = r'(?:id|artifact-file-id)\s+([0-9a-f]{32})'
            match = re.search(pattern, query)

            if match:
                return match.group(1)
            return None
        except Exception:
            return None

    def invoke(self, query, session_id) -> str:
        """Kickoff CrewAI and return the response."""
        artifact_file_id = self.extract_artifact_file_id(query)

        inputs = {
            'user_topic': query,
            'session_id': session_id,
            'artifact_file_id': artifact_file_id,
        }
        logger.info(f'Inputs {inputs}')
        print(f'Inputs {inputs}')
        response = self.crew.kickoff(inputs)
        return response

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')

    def get_summary(self, result) -> str:
        """Return the summary text from the result."""
        return result.raw if hasattr(result, 'raw') else str(result)
