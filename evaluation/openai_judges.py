"""OpenAI-backed LLM judges for the evaluation harness."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from dotenv import load_dotenv

DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.4-mini"

__all__ = [
    "DEFAULT_OPENAI_JUDGE_MODEL",
    "OpenAIJudgeClient",
    "build_openai_judge_fns",
]


class OpenAIJudgeClient:
    """Thin wrapper around the OpenAI Responses API for structured judging."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str | None = None,
        max_output_tokens: int = 1500,
    ) -> None:
        load_dotenv()
        self.model = model or os.environ.get("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_JUDGE_MODEL
        self.api_key = api_key or os.environ.get(api_key_env)
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if not self.api_key:
            raise ValueError(f"OpenAI judging requires {api_key_env} or an explicit API key.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("OpenAI judging requires the openai Python package.") from exc

        self._client = OpenAI(api_key=self.api_key)

    def judge_paper_relevance(self, task: dict[str, Any]) -> dict[str, Any]:
        """Judge one query-paper relevance task."""
        return self._structured_response(
            instructions=_paper_relevance_instructions(),
            task=_paper_relevance_prompt_payload(task),
            schema=_paper_relevance_schema(),
            schema_name="paper_relevance_judgment",
        )

    def judge_exploration_range(self, task: dict[str, Any]) -> dict[str, Any]:
        """Judge broad-query result-list exploration range."""
        return self._structured_response(
            instructions=_exploration_range_instructions(),
            task=_exploration_range_prompt_payload(task),
            schema=_exploration_range_schema(),
            schema_name="exploration_range_judgment",
        )

    def judge_win_rate(self, task: dict[str, Any]) -> dict[str, Any]:
        """Judge the best anonymous result set for one query."""
        labels = sorted(task.get("systems", {}))
        return self._structured_response(
            instructions=_win_rate_instructions(labels),
            task=_win_rate_prompt_payload(task),
            schema=_win_rate_schema(labels),
            schema_name="anonymous_win_rate_judgment",
        )

    def _structured_response(
        self,
        *,
        instructions: str,
        task: dict[str, Any],
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Call the Responses API and parse a schema-constrained JSON response."""
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(task, ensure_ascii=False, sort_keys=True),
                        }
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        response = self._client.responses.create(**request)
        text = _extract_response_text(response)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI judge returned invalid JSON: {text!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI judge returned a non-object JSON response.")
        return parsed


def build_openai_judge_fns(
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    reasoning_effort: str | None = None,
    max_output_tokens: int = 1500,
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], ...]:
    """Build paper, range, and win-rate judge callbacks."""
    client = OpenAIJudgeClient(
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    return (
        client.judge_paper_relevance,
        client.judge_exploration_range,
        client.judge_win_rate,
    )


def _paper_relevance_instructions() -> str:
    return (
        "You are an impartial evaluator for a closed-corpus academic paper search system. "
        "Judge whether the paper is useful for the query using only the provided title, "
        "abstract, authors, year, and metadata. Do not reward famous papers unless they "
        "fit the query. For specific queries, emphasize exact technical fit. For broad "
        "queries, judge topical usefulness but leave list diversity to the separate range judge."
    )


def _exploration_range_instructions() -> str:
    return (
        "You are an impartial evaluator for broad academic literature search. Judge whether "
        "the top-10 papers cover meaningfully different useful subtopics within the broad "
        "query while staying on topic. Do not reward arbitrary diversity or off-topic variety."
    )


def _win_rate_instructions(labels: list[str]) -> str:
    label_text = ", ".join(labels)
    return (
        "You are an impartial evaluator comparing anonymous academic search result sets. "
        f"The systems are anonymized as {label_text}. Do not infer model identity. Use the "
        "support metrics as evidence, but make the final decision by considering both the "
        "metrics and the actual returned papers. Prefer result sets that are relevant, well "
        "ranked, cover the judged pooled relevant set, and provide useful topical breadth. "
        "Return pairwise preferences for every supplied pair."
    )


def _paper_relevance_prompt_payload(task: dict[str, Any]) -> dict[str, Any]:
    paper = dict(task.get("paper") or {})
    return {
        "query_id": task.get("query_id"),
        "query_text": task.get("query_text"),
        "query_type": task.get("query_type"),
        "paper": _trim_paper_payload(paper),
        "rubric": {
            "0": "irrelevant",
            "1": "loosely related but not a strong answer",
            "2": "relevant and useful",
            "3": "highly relevant or strong fit",
        },
    }


def _exploration_range_prompt_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": task.get("query_id"),
        "query_text": task.get("query_text"),
        "papers": [_trim_paper_payload(paper) for paper in task.get("papers", [])],
        "rubric": {
            "1": "very narrow",
            "2": "limited range",
            "3": "moderate range",
            "4": "good range",
            "5": "strong range across meaningful subtopics",
        },
    }


def _win_rate_prompt_payload(task: dict[str, Any]) -> dict[str, Any]:
    systems = {}
    for label, payload in sorted((task.get("systems") or {}).items()):
        systems[label] = {
            "support_metrics": payload.get("support_metrics") or {},
            "results": [
                _trim_paper_payload(paper)
                for paper in (payload.get("results") or [])
            ],
        }
    return {
        "query_id": task.get("query_id"),
        "query_text": task.get("query_text"),
        "query_type": task.get("query_type"),
        "metrics_used": task.get("metrics_used"),
        "systems": systems,
    }


def _trim_paper_payload(paper: dict[str, Any]) -> dict[str, Any]:
    """Keep judge prompts bounded while preserving the important evidence."""
    abstract = str(paper.get("abstract") or "")
    if len(abstract) > 1800:
        abstract = abstract[:1800].rsplit(" ", 1)[0] + "..."
    return {
        "rank": paper.get("rank"),
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "abstract": abstract,
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "fields_of_study": paper.get("fields_of_study") or paper.get("categories") or [],
        "subfields": paper.get("subfields") or [],
        "citation_count": paper.get("citation_count"),
        "reference_count": paper.get("reference_count"),
    }


def _paper_relevance_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relevance_score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "short_rationale": {"type": "string"},
        },
        "required": ["relevance_score", "short_rationale"],
    }


def _exploration_range_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "exploration_range_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "represented_subtopics": {
                "type": "array",
                "items": {"type": "string"},
            },
            "short_rationale": {"type": "string"},
        },
        "required": [
            "exploration_range_score",
            "represented_subtopics",
            "short_rationale",
        ],
    }


def _win_rate_schema(labels: list[str]) -> dict[str, Any]:
    winner_values = [*labels, "tie", "no_clear_winner"]
    preference_values = [*labels, "tie", "no_preference"]
    pairwise_properties = {
        f"{left}_vs_{right}": {"type": "string", "enum": preference_values}
        for left_index, left in enumerate(labels)
        for right in labels[left_index + 1 :]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "winner": {"type": "string", "enum": winner_values},
            "tied_systems": {
                "type": "array",
                "items": {"type": "string", "enum": labels},
            },
            "ranked_systems": {
                "type": "array",
                "items": {"type": "string", "enum": labels},
            },
            "pairwise_preferences": {
                "type": "object",
                "additionalProperties": False,
                "properties": pairwise_properties,
                "required": list(pairwise_properties),
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "short_rationale": {"type": "string"},
        },
        "required": [
            "winner",
            "tied_systems",
            "ranked_systems",
            "pairwise_preferences",
            "confidence",
            "short_rationale",
        ],
    }


def _extract_response_text(response: Any) -> str:
    """Extract text from current and older OpenAI SDK response objects."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for content_item in content:
                text = getattr(content_item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    raise ValueError("OpenAI response did not include output text.")
