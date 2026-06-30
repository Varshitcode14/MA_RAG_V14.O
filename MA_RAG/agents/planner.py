import ast
import json
import re

from utils.provider_manager import ProviderManager

from MA_RAG.prompts.planner_prompt import PLANNER_PROMPT


provider = ProviderManager()


def _parse_plan(response: str, question: str) -> list[str]:
    """
    Robustly parse the planner's output into a list of step strings.

    The LLM is asked for a list of strings, but models often wrap it in
    prose, code fences, or use single/double quotes inconsistently.
    We try, in order:
      1. Extract the first [...] block and parse as JSON.
      2. Parse that block with ast.literal_eval (handles single quotes).
      3. Fall back to the original question as a single step.
    """
    if not response:
        return [question]

    text = response.strip()

    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    # Grab the first bracketed list in the text.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text

    for parser in (json.loads, ast.literal_eval):
        try:
            plan = parser(candidate)
            if isinstance(plan, list):
                steps = [str(s).strip() for s in plan if str(s).strip()]
                if steps:
                    return steps[:3]  # planner contract: never more than 3 steps
        except Exception:
            continue

    return [question]


def planner_agent(state):

    prompt = PLANNER_PROMPT.format(question=state["question"])

    response = provider.generate(prompt)

    state["plan"] = _parse_plan(response, state["question"])
    state["current_step"] = 0

    return state
