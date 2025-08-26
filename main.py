"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging
import os

import click

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent import FinanceResearchAgent
from agent_executor import FinanceResearchAgentExecutor
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='0.0.0.0')
@click.option('--port', 'port', default=8080)
def main(host, port):
    """Entry point for the A2A + CrewAI Image generation sample."""
    try:
        if not os.getenv('GOOGLE_API_KEY') and not os.getenv(
            'GOOGLE_GENAI_USE_VERTEXAI'
        ):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY or Vertex AI environment variables not set.'
            )

        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
        id='financial_researcher',
        name='Financial Researcher',
        description=(
            'Research any topic and provide a summary focused on financial aspects, trends, and impacts.'
        ),
        tags=['finance', 'research', 'web search', 'scraping'],
        examples=['Summarize the financial impact of electric vehicles'],
        )

        agent_host_url = (
            os.getenv('HOST_OVERRIDE')
            if os.getenv('HOST_OVERRIDE')
            else f'http://{host}:{port}/'
        )
        agent_card = AgentCard(
            name='Financial Researcher',
            description=(
                'FinancialResearcher is a web-powered analytical agent designed to extract and summarize the financial dimensions of any given topic. The agent performs real-time web searches and scraping to gather data from trusted sources, then distills key insights into a concise financial summary. Ideal for use cases like market trend analysis, fiscal impact assessments, or investment research.'
            ),
            url=f"http://localhost:{port}/",
            version='1.0.0',
            default_input_modes=FinanceResearchAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=FinanceResearchAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=FinanceResearchAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        import uvicorn

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
