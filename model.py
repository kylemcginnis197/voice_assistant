import anthropic
import chromadb
import inspect
import json

from pydantic import BaseModel, Field
from log import get_logger
from config import (
    TOOL_EMBEDDINGS_RAG,
    TOOL_RAG_TOP_K
)
import time

log = get_logger("model")

def get_pydantic_parameters(tool: object):
    """Returns the first parameter's annotation if the tool has parameters, else None."""
    sig = inspect.signature(obj=tool)

    if len(sig.parameters) == 0:
        return None

    sig_iter = sig.parameters.items().__iter__()
    _, parameters = next(sig_iter)
    return parameters

def generate_declarations(tools: list[object]) -> list[dict]:
    """Auto-generate Anthropic tool declarations from tool functions."""
    res = []

    for tool in tools:
        name = tool.__name__

        if not name:
            raise RuntimeError("Failed to get function name during declaration generation.")

        tool_declaration = {
            "name": tool.__name__
        }

        if description := tool.__doc__:
            tool_declaration["description"] = description

        if parameter := get_pydantic_parameters(tool):
            tool_schema = parameter.annotation.model_json_schema()

            if tool_schema:
                tool_declaration["input_schema"] = {
                    "type": "object",
                    "properties": tool_schema.get("properties", {}),
                    "required": tool_schema.get("required", []),
                }

        res.append(tool_declaration)

    return res

class ModelOutput(BaseModel):
    speech: str = Field(description="The final message that will be spoken to the user.")

class ToolRetrieval:
    def __init__(self, tools: list[dict]):
        # ChromaDb in memory
        self.client = chromadb.Client() 

        self.collection = self.client.get_or_create_collection(
            name = "tools",
            metadata={"hnsw:space": "cosine"}
        )

        self.collection.upsert(
            ids = [t["name"] for t in tools],
            documents=[t.get("description", t["name"]) for t in tools],
            metadatas=[{"schema": json.dumps(t)} for t in tools],  # store full def
        )

    def retrieve_tools(self, query: list[str], top_k: int = 4) -> list[dict]:
        results = self.collection.query(query_texts=query, n_results=top_k)
        return [json.loads(tool["schema"]) for tool in results["metadatas"][0]]


class Model:
    def __init__(self, tools: list[object], always_included_tools: list[object], web_search: bool = True) -> None:
        self.client = anthropic.Client()
        self.input_token_limit = 75_000
        self.output_token_limit = 4_096

        # model options
        self.beta_supported = ["claude-opus-4-6", "claude-sonnet-4-6"]
        self.thinking_model = "claude-sonnet-4-6-20250929"

        # create context window
        self.context_window = []

        # generate tool declarations
        self.tool_references = tools

        if TOOL_EMBEDDINGS_RAG:
            self.tool_retrieval = ToolRetrieval(tools=generate_declarations(tools))
        else:
            self.tools = generate_declarations(tools)

        # Tools always sent to the model regardless of RAG retrieval
        self.always_include_tools = generate_declarations(always_included_tools)

        if web_search:
            self.always_include_tools.append({
                "type": "web_search_20260209",
                "name": "web_search",
                "user_location": {
                    "type": "approximate",
                    "city": "Omaha",
                    "region": "Nebraska",
                    "country": "US",
                    "timezone": "America/Chicago"
                }
            })

        # define output schema.
        schema = ModelOutput.model_json_schema()
        schema["additionalProperties"] = False

        self.output_schema = {
            "format": {
                "type": "json_schema",
                "schema": schema
            }
        }

        try:
            with open(file="prompts/system_prompt.md", mode="r") as file:
                contents = file.read()
                self.system_prompt = contents
        except Exception as e:
            log.error(f"Failed to load system prompt: {e}")
            self.system_prompt = None

    def clear_context_window(self):
        self.context_window = []

    def set_input_tokens(self, tokens):
        self.input_token_limit = tokens

    def set_output_tokens(self, tokens):
        self.output_token_limit = tokens

    def set_model(self, model_name: str) -> None:
        model_name = model_name.lower().strip()

        if model_name == "opus 4.6":
            self.thinking_model = "claude-opus-4-6"
        elif model_name == "sonnet 4.6":
            self.thinking_model = "claude-sonnet-4-6"
        elif model_name == "sonnet 4.5":
            self.thinking_model = "claude-sonnet-4-5-20250929"
        else:
            self.thinking_model = "claude-sonnet-4-6"

    async def execute_tool(self, tool_name, tool_args):
        for tool in self.tool_references:
            if tool.__name__ == tool_name:
                parameter = get_pydantic_parameters(tool)

                log.info(f"Tool call: {tool.__name__} with args: {tool_args}")

                try:
                    if parameter and parameter.annotation is not inspect.Parameter.empty:
                        pydantic_class = parameter.annotation
                        response = await tool(pydantic_class(**tool_args)) if inspect.iscoroutinefunction(tool) else tool(pydantic_class(**tool_args))
                    else:
                        response = await tool() if inspect.iscoroutinefunction(tool) else tool()
                except Exception as e:
                    return {
                        "status": "error",
                        "response": f"Failed to run {tool_name} with args: {tool_args}, error: {e}"
                    }
                else:
                    return {
                        "status": "success",
                        "response": response
                    }
        return {
            "status": "error",
            "response": f"Failed to find tool. No tool named {tool_name} with args: {tool_args}"
        }

    async def call_model(self, input: str, ts) -> str:
        self.context_window.append({
            "role": "user",
            "content": input
        })

        while True:
            args = {}

            args["model"] = self.thinking_model
            args["max_tokens"] = self.output_token_limit
            args["system"] = self.system_prompt

            if TOOL_EMBEDDINGS_RAG:
                last_couple_user_requests = [context["content"] for context in self.context_window if context["role"] == "user" and isinstance(context["content"], str)]

                if len(last_couple_user_requests) > 2:
                    last_couple_user_requests = last_couple_user_requests[-2:]

                args["tools"] = self.tool_retrieval.retrieve_tools(query=last_couple_user_requests, top_k=TOOL_RAG_TOP_K) + self.always_include_tools

                print(f"[model] Retrieved Tools: {[tool.get('name') for tool in args['tools'] if tool.get('name', None) is not None]}")
            else:
                args["tools"] = self.tools + self.always_include_tools

            args["messages"] = self.context_window
            args["output_config"] = self.output_schema

            # Compaction only supported on opus 4.6 or sonnet 4.6   
            if args["model"] in self.beta_supported:
                args["betas"] = ["compact-2026-01-12"]
                args["thinking"] = {"type": "adaptive"} # adaptive, enabled, or disabled (adaptive + enabled only available on opus 4.6 or sonnet 4.6)
                args["context_management"] = {
                        "edits": [{
                            "type": "compact_20260112",
                            "trigger": {"type": "input_tokens", "value": self.input_token_limit},
                            "pause_after_compaction": True,
                            "instructions": "Focus on preserving big picture ideas and themes and tool call responses"
                        }]
                    }


            # Call model.
            print(f"[model] Called model ({time.monotonic() - ts:.2f}s)")

            with self.client.beta.messages.stream(**args) as stream:
                response = stream.get_final_message()

            print(f"[model] Response generated ({time.monotonic() - ts:.2f}s)")

            stop_reason = response.stop_reason
            content = response.content

            if stop_reason == "end_turn" or stop_reason == "max_tokens":
                obj = None

                for block in content:
                    if block.type == "text":
                        try:
                            obj = json.loads(block.text)
                        except Exception as e:
                            print(f"[model] Model returned incorrect json: {e}")
                
                output = obj.get("speech", None)
                return output
            elif stop_reason == "tool_use":
                self.context_window.append({"role": "assistant", "content": content})

                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        res = await self.execute_tool(tool_name=block.name, tool_args=block.input)

                        if res is None or not len(res):
                            res = "tool ran successfully."

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(res)
                        })

                self.context_window.append({
                    "role": "user",
                    "content": tool_results
                })
            elif stop_reason == "compaction":
                compacted_block = content[0]
                preserved_messages = self.context_window[-4:] if len(self.context_window) > 4 else self.context_window
                self.context_window = [compacted_block] + preserved_messages
            elif stop_reason == "pause_turn":
                self.context_window.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Please continue."}
                ])
            elif stop_reason == "refusal":
                return "I'm not answering that shit."
