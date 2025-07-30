# user_satisfaction_survey_agent/__init__.py
import logging
import pathlib
import textwrap
from dataclasses import asdict

import agents
import jinja2
import openai
import pydantic
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
    ) -> "UserSatisfactionSurveyResult":
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

        return UserSatisfactionSurveyResult(
            messages=messages,
            usage=usage,
            sentiment_metrics=self._parse_sentiment_metrics(result.final_output),
            customer_effort_metrics=self._parse_customer_effort_metrics(
                result.final_output
            ),
            semantic_cohesion_metric=self._parse_semantic_cohesion_metric(
                result.final_output
            ),
        )

    def _parse_sentiment_metrics(self, text: str) -> "SentimentMetrics":
        return SentimentMetrics.model_validate(text)

    def _parse_customer_effort_metrics(self, text: str) -> "CustomerEffortMetrics":
        return CustomerEffortMetrics.model_validate(text)

    def _parse_semantic_cohesion_metric(self, text: str) -> "SemanticCohesionMetric":
        return SemanticCohesionMetric.model_validate(text)

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


class SentimentMetrics(pydantic.BaseModel):
    average_sentiment_score: float = pydantic.Field(
        description="The average sentiment value (e.g., from -1 to +1) calculated across all of the customer's messages in a conversation."  # noqa: E501
    )
    sentiment_trend_slope: float = pydantic.Field(
        description="A value calculated via linear regression that shows whether the customer's sentiment improved (positive slope) or worsened (negative slope) during the conversation."  # noqa: E501
    )
    final_sentiment_state: float = pydantic.Field(
        description="The average sentiment score of the customer's final few messages, reflecting their concluding impression of the service."  # noqa: E501
    )


class CustomerEffortMetrics(pydantic.BaseModel):
    frequency_of_repeated_intent: int = pydantic.Field(
        description="A count of how many times the customer had to repeat the same question or express the same need, indicating the issue was not resolved on the first attempt."  # noqa: E501
    )
    negative_signal_word_frequency: int = pydantic.Field(
        description="A count of specific keywords used by the customer that imply high effort or frustration (e.g., 'again,' 'still,' 'I already told you,' 'useless')."  # noqa: E501
    )
    question_density: float = pydantic.Field(
        description="The percentage of the customer's messages that are questions, which can indicate confusion if the ratio is unusually high."  # noqa: E501
    )


class SemanticCohesionMetric(pydantic.BaseModel):
    question_answer_cosine_similarity: float = pydantic.Field(
        description="A score representing how semantically relevant the agent's answer is to the customer's question, calculated by comparing their vector embeddings. A higher score means a more relevant answer."  # noqa: E501
    )


class UserSatisfactionSurveyResult(pydantic.BaseModel):
    messages: list["Message"]
    usage: Usage
    sentiment_metrics: SentimentMetrics = pydantic.Field(
        description="The sentiment metrics of the customer's conversation."
    )
    customer_effort_metrics: CustomerEffortMetrics = pydantic.Field(
        description="The customer effort metrics of the customer's conversation."
    )
    semantic_cohesion_metric: SemanticCohesionMetric = pydantic.Field(
        description="The semantic cohesion metric of the customer's conversation."
    )
