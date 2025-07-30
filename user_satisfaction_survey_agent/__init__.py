# user_satisfaction_survey_agent/__init__.py
import logging
import pathlib
import textwrap
from dataclasses import asdict

import agents
import jinja2
import openai
import rich.console
import rich.panel
from openai.types import ChatModel
from rich_color_support import RichColorRotator

from user_satisfaction_survey_agent._message import Message
from user_satisfaction_survey_agent._usage import Usage

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)
console = rich.console.Console()
color_rotator = RichColorRotator()

DEFAULT_MODEL = "gpt-4.1-nano"


class UserSatisfactionSurveyAgent:
    instructions: str = textwrap.dedent(
        """
        """  # noqa: E501
    )

    async def analyze_rules_and_memories(
        self,
        messages: list["Message"],
        *,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
        tracing_disabled: bool = True,
        verbose: bool = False,
        console: rich.console.Console = console,
        color_rotator: RichColorRotator = color_rotator,
        width: int = 80,
        **kwargs,
    ) -> dict:
        chat_model = self._to_chat_model(model)

        agent_instructions_template = jinja2.Template(self.instructions)
        user_input = agent_instructions_template.render()

        if verbose:
            __rich_panel = rich.panel.Panel(
                user_input,
                title="LLM INSTRUCTIONS",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)

        agent = agents.Agent(
            name="user-preferences-agent-rules-and-memories",
            model=chat_model,
            model_settings=agents.ModelSettings(temperature=0.0),
        )
        result = await agents.Runner.run(
            agent,
            user_input,
            run_config=agents.RunConfig(tracing_disabled=tracing_disabled),
        )
        usage = Usage.model_validate(asdict(result.context_wrapper.usage))

        if verbose:
            __rich_panel = rich.panel.Panel(
                str(result.final_output),
                title="LLM OUTPUT",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)
            __rich_panel = rich.panel.Panel(
                usage.model_dump_json(indent=4),
                title="LLM USAGE",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)

        return dict(
            messages=messages,
            result=result.final_output,
            usage=usage,
        )

    def _to_chat_model(
        self,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
    ) -> agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel:
        model = DEFAULT_MODEL if model is None else model

        if isinstance(model, str):
            openai_client = openai.AsyncOpenAI()
            return agents.OpenAIResponsesModel(
                model=model,
                openai_client=openai_client,
            )

        else:
            return model
