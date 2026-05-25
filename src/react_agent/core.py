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
        self.current_task_type = None

    def detect_task_type(self, request: str) -> str:
        """基于关键词检测任务类型"""
        request_lower = request.lower()
        
        simple_patterns = [
            '有多少', '多少个', 'how many', '是否存在', '有没有',
            '存在吗', 'count', 'how much', '基因数', '样本数',
            '几个基因', '几个样本'
        ]
        if any(p in request_lower for p in simple_patterns):
            return 'simple_qa'
        
        viz_patterns = ['绘制', '创建', 'plot', 'create', 'draw', '生成', '条形图', '热图', '散点图']
        if any(p in request_lower for p in viz_patterns):
            return 'visualization'
        
        analysis_patterns = ['分析', 'analyze', 'GO', 'enrichment', '富集']
        if any(p in request_lower for p in analysis_patterns):
            return 'custom_analysis'
        
        return 'unknown'

    def is_task_complete(self, task_type: str, tool_name: str, tool_result: Any) -> bool:
        """检测任务是否已完成"""
        if not isinstance(tool_result, dict):
            return False
        
        if task_type == 'simple_qa' and tool_name == 'get_data_info':
            return tool_result.get('success', False)
        
        if tool_result.get('saved_path'):
            return True
        
        if tool_name == 'execute_code' and tool_result.get('success'):
            return True
        
        return False

    def extract_answer_from_result(self, task_type: str, tool_name: str, 
                                  tool_result: Any, user_request: str) -> str:
        """从工具结果中提取答案"""
        if not isinstance(tool_result, dict):
            return str(tool_result)
        
        result = tool_result.get('result', tool_result)
        
        if task_type == 'simple_qa' and tool_name == 'get_data_info':
            info = tool_result.get('info', {})
            if not info and 'result' in tool_result:
                try:
                    info = tool_result['result'].get('info', {})
                except:
                    pass
            
            shape = info.get('shape', ())
            if len(shape) >= 2:
                genes_count = shape[0]
                samples_count = shape[1] if len(shape) > 1 else 0
                
                request_lower = user_request.lower()
                if '基因' in request_lower or 'gene' in request_lower:
                    if '样本' in request_lower or 'sample' in request_lower:
                        return f"您的表达谱数据包含{genes_count}个基因和{samples_count}个样本。"
                    return f"您的表达谱数据包含{genes_count}个基因。"
                elif '样本' in request_lower or 'sample' in request_lower:
                    return f"您的表达谱数据包含{samples_count}个样本。"
        
        if 'message' in tool_result:
            return tool_result['message']
        
        return f"Task completed: {tool_result.get('message', 'Success')}"

    def should_auto_terminate(self, task_type: str, step_count: int,
                             last_tool: str, last_result: Any) -> tuple[bool, Optional[str]]:
        """判断是否应该自动终止"""
        if last_result is None:
            return False, None
        
        if not isinstance(last_result, dict):
            return False, None
        
        if task_type == 'simple_qa':
            if step_count >= 2 and last_tool == 'get_data_info':
                if last_result.get('success'):
                    answer = self.extract_answer_from_result(
                        task_type, last_tool, last_result, 
                        getattr(self, 'user_request', '')
                    )
                    return True, answer
        
        if last_result.get('saved_path'):
            answer = f"Visualization saved to {last_result['saved_path']}"
            return True, answer
        
        if last_tool == 'execute_code' and last_result.get('success'):
            answer = self.extract_answer_from_result(task_type, last_tool, last_result, '')
            return True, answer
        
        if task_type == 'simple_qa' and last_tool == 'get_data_info':
            if last_result.get('success'):
                answer = self.extract_answer_from_result(
                    'simple_qa', last_tool, last_result, 
                    getattr(self, 'user_request', '')
                )
                return True, answer
        
        return False, None

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
        
        self.user_request = user_input
        self.current_task_type = self.detect_task_type(user_input)
        
        if self.verbose:
            print(f"\n[Auto-Detect] Task type: {self.current_task_type}")
        
        last_tool_result = None
        last_tool_name = None
        
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
                last_tool_name = action_name
                last_tool_result = observation.result
                
                if self.verbose:
                    if observation.success:
                        print(f"[Observation] Success: {str(observation.result)[:200]}...")
                    else:
                        print(f"[Observation] Failed: {observation.error}")
            
            self.history.append(step)
            
            auto_terminate, auto_answer = self.should_auto_terminate(
                self.current_task_type,
                len(self.history),
                last_tool_name,
                last_tool_result
            )
            
            if auto_terminate and auto_answer:
                if self.verbose:
                    print(f"\n[Auto-Terminate] Task complete, generating final_answer automatically")
                    print(f"[Final Answer] {auto_answer}")
                
                return {
                    "success": True,
                    "answer": {"message": auto_answer},
                    "steps": [s.to_dict() for s in self.history]
                }
            
            if not action_name:
                break
        
        return {
            "success": False,
            "answer": "Maximum iterations reached or process ended without final answer",
            "steps": [s.to_dict() for s in self.history]
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.history]
