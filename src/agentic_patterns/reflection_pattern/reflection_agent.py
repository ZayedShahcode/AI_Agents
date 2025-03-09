from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import FixedFirstChatHistory
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()

BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with revised version of your previous attempt.
You must always output the revised version but don't mention this is a revised version.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generatd content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change output this: <OK>
"""

class ReflectionAgent:
    """
    Attributes:
        model(str): The model used for generating and reflecting on responses.
        client(Groq): An instance of the Groq client to interact with language model.
    """

    def __init__(self,model: str = "llama3-70b-8192"):
        self.client = Groq()
        self.model = model
        self.generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=BASE_GENERATION_SYSTEM_PROMPT,role="system"),
            ],
            total_length=10,
        )
        self.reflection_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=BASE_REFLECTION_SYSTEM_PROMPT, role="system"),
            ],
            total_length=10,
        )
        
    
    def _request_completion(
            self,
            history: list,
            verbose: int=0,
            log_title: str = "COMPLETION",
            log_color: str=""
    ):
        "A method to request completion (response) from model."
        output = completions_create(self.client,history,self.model)
        return output
    
    def generate(self,generation_history : list,verbose: int=0) -> str:
        return self._request_completion(generation_history,verbose,log_title="GENERATION",log_color=Fore.BLUE)
    
    def reflect(self,reflection_history: list,verbose: int=0) -> str:
        return self._request_completion(reflection_history,verbose,log_title="REFLECTION",log_color=Fore.GREEN)
    
    def run(
        self,
        user_msg: str,
        n_steps: int = 10,
        verbose: int = 0,
) -> str:
        """Runs the agent to generate responses and refine them iteratively."""

        user_input = build_prompt_structure(prompt=user_msg, role="user")

        if isinstance(user_input, dict) and "content" in user_input:
            user_input = user_input["content"]

        update_chat_history(self.generation_history, user_input, "user")

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

            generation = self.generate(self.generation_history,verbose=verbose)

            if not isinstance(generation, str):
                raise ValueError(f"Expected string, but got {type(generation)}: {generation}")

            update_chat_history(self.generation_history, generation, "assistant")
            update_chat_history(self.reflection_history, generation, "user")

            critique = self.reflect(self.reflection_history,verbose=verbose)

            if "<OK>" in critique:
                print(
                    Fore.RED,
                    "\n\n Stop Sequence found. Stopping the reflection loop... \n\n"
                )
                break

            if not isinstance(critique, str):
                raise ValueError(f"Expected string, but got {type(critique)}: {critique}")

            update_chat_history(self.generation_history, critique, "user")
            update_chat_history(self.reflection_history, critique, "assistant")

        return generation
