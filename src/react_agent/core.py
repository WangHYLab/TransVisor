from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
import time
from datetime import datetime


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


@dataclass
class Thought:
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": StepType.THOUGHT.value,
            "content": self.content,
            "timestamp": self.timestamp
        }


@dataclass
class Action:
    tool_name: str
    tool_input: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": StepType.ACTION.value,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "timestamp": self.timestamp
        }


@dataclass
class Observation:
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": StepType.OBSERVATION.value,
            "tool_name": self.tool_name,
            "result": str(self.result)[:500] if self.result else None,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp
        }


@dataclass
class ReActStep:
    thought: Thought
    action: Optional[Action] = None
    observation: Optional[Observation] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "thought": self.thought.to_dict()
        }
        if self.action:
            result["action"] = self.action.to_dict()
        if self.observation:
            result["observation"] = self.observation.to_dict()
        return result


class ReActAgent:
    def __init__(
        self,
        model_client: Any,
        tools: Dict[str, Callable],
        system_prompt: str,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.model_client = model_client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history: List[ReActStep] = []

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]

        for step in self.history:
            step_dict = step.to_dict()
            messages.append({
                "role": "user",
                "content": f"Thought: {step_dict['thought']['content']}"
            })
            if step.action:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "action": step_dict['action']['tool_name'],
                        "input": step_dict['action']['tool_input']
                    })
                })
            if step.observation:
                obs = step_dict['observation']
                content = f"Observation: {obs['result']}"
                if not obs['success']:
                    content += f"\nError: {obs['error']}"
                messages.append({
                    "role": "user",
                    "content": content
                })

        messages.append({"role": "user", "content": user_input})
        return messages

    def _parse_response(self, response: str) -> tuple[str, Optional[str], Optional[Dict]]:
        try:
            parsed = json.loads(response)
            thought = parsed.get("thought", "")
            action_name = parsed.get("action")
            action_input = parsed.get("input", {})
            return thought, action_name, {"input": action_input}
        except:
            return response, None, None

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Observation:
        if tool_name not in self.tools:
            return Observation(
                tool_name=tool_name,
                result=None,
                success=False,
                error=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            )

        try:
            result = self.tools[tool_name](**tool_input)
            return Observation(tool_name=tool_name, result=result, success=True)
        except Exception as e:
            return Observation(
                tool_name=tool_name,
                result=None,
                success=False,
                error=str(e)
            )

    def run(self, user_input: str, clear_history: bool = False) -> Dict[str, Any]:
        if clear_history:
            self.history = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print('='*60)

            messages = self._build_messages(user_input)

            try:
                response = self.model_client.chat(messages)
            except Exception as e:
                if self.verbose:
                    print(f"Model call failed: {e}")
                break

            thought_content, action_name, action_input = self._parse_response(response)

            if self.verbose:
                print(f"\n[Thought] {thought_content}")
                if action_name:
                    print(f"[Action] {action_name}: {action_input}")

            thought = Thought(content=thought_content)
            step = ReActStep(thought=thought)

            if action_name and action_name.lower() == "final_answer":
                if self.verbose:
                    print(f"\n[Final Answer] {action_input}")
                return {
                    "success": True,
                    "answer": action_input.get("answer", str(action_input)) if isinstance(action_input, dict) else str(action_input),
                    "steps": [s.to_dict() for s in self.history]
                }

            if action_name:
                observation = self._execute_tool(action_name, action_input.get("input", {}))
                step.action = Action(tool_name=action_name, tool_input=action_input.get("input", {}))
                step.observation = observation

                if self.verbose:
                    if observation.success:
                        print(f"[Observation] Success: {str(observation.result)[:200]}...")
                    else:
                        print(f"[Observation] Failed: {observation.error}")

            self.history.append(step)

            if not action_name:
                break

        return {
            "success": False,
            "answer": "Maximum iterations reached or process ended without final answer",
            "steps": [s.to_dict() for s in self.history]
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.history]
