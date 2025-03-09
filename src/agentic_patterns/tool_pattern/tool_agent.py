import json
import re

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.tool_pattern.tool import validate_arguments
from src.agentic_patterns.utils.completions import build_prompt_structure
from src.agentic_patterns.utils.completions import ChatHistory
from src.agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.extraction import extract_tag_content

load_dotenv()

TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""

class ToolAgent:
    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama3-70b-8192"
    )->None:
        self.client =Groq()
        self.model = model
        self.tools = tools if isinstance(tools,list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system"
                ),
            ]
        )
        self.agent_chat_history = ChatHistory([])
    
    def add_tool_signatures(self)->str:
        return "".join([tool.fn_signature for tool in self.tools])
    
    def process_tool_calls(self,tool_calls_content:list)->dict:
        observations={}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call['name']
            tool = self.tools_dict[tool_name]
            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + "\nTool Result: \n{result}")

            observations[validated_tool_call["id"]] = result
        return observations
    
    def run(
        self,
        user_msg:str,
    )->str:
        user_prompt = build_prompt_structure(prompt=user_msg,role="user")
        update_chat_history(self.tool_chat_history,user_prompt,"user")
        update_chat_history(self.agent_chat_history,user_prompt,"user")

        print(self.tool_chat_history)
        tool_call_response = completions_create(
            self.client,messages=self.tool_chat_history, model=self.model
        )
        tool_calls = extract_tag_content(str(tool_call_response),"tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(
                self.agent_chat_history,f'f"Observation: {observations}"',"user"
            )
        return completions_create(self.client,self.agent_chat_history,self.model)