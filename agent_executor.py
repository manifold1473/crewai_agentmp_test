from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    FilePart,
    FileWithBytes,
    InvalidParamsError,
    Part,
    Task,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    completed_task,
    new_artifact,
)
from a2a.utils.errors import ServerError
from agent import FinanceResearchAgent


class FinanceResearchAgentExecutor(AgentExecutor):
    """Finance Research AgentExecutor."""

    def __init__(self):
        self.agent = FinanceResearchAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        try:
            result = self.agent.invoke(query, context.context_id)
            print(f'Final Result ===> {result}')
        except Exception as e:
            print('Error invoking agent: %s', e)
            raise ServerError(
                error=ValueError(f'Error invoking agent: {e}')
            ) from e

        summary = self.agent.get_summary(result)
        parts = [
            Part(
                root=TextPart(text=summary)
            )
        ]
        await event_queue.enqueue_event(
            completed_task(
                context.task_id,
                context.context_id,
                [new_artifact(parts, f'finance_summary_{context.task_id}')],
                [context.message],
            )
        )

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        return False
