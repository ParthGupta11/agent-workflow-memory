import os
import dataclasses

from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.python import PythonActionSet
from browsergym.utils.obs import flatten_axtree_to_str


class DemoAgent(Agent):
    """A basic agent using an OpenAI-compatible API (OpenAI, Gemini, Self-hosted)."""

    action_set = HighLevelActionSet(
        subsets=["chat", "bid"],
        strict=False,
        multiaction=True,
        demo_mode="default",
    )

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "goal": obs["goal"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        }

    def __init__(
        self, model_name: str, base_url: str = None, api_key: str = None
    ) -> None:
        super().__init__()
        self.model_name = model_name

        from openai import OpenAI

        # Initialize the client with optional overrides for Gemini or self-hosted models
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)

    def get_action(self, obs: dict) -> tuple[str, dict]:
        system_msg = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{obs["goal"]}"""

        prompt = f"""\
# Current Accessibility Tree:
{obs["axtree_txt"]}

# Action Space
{self.action_set.describe(with_long_description=False, with_examples=True)}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
"""

        # query the model
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        action = response.choices[0].message.content

        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    Stores the arguments that define the agent.
    """

    model_name: str = "gemini-2.0-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = None

    def make_agent(self):
        # Fall back to GEMINI_API_KEY environment variable if not explicitly passed
        key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        return DemoAgent(
            model_name=self.model_name, base_url=self.base_url, api_key=key
        )


def main():
    from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
    from pathlib import Path

    exp_root = Path().home() / "agent_experiments"
    exp_root.mkdir(exist_ok=True)

    exp_args = ExpArgs(
        agent_args=DemoAgentArgs(
            model_name="gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ.get("GEMINI_API_KEY"),
        ),
        env_args=EnvArgs(
            task_name="miniwob.click-test",
            task_seed=42,
            headless=False,
        ),
    )

    exp_args.prepare(exp_root=exp_root)
    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
