import io
import sys
import traceback
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np


class CodeExecutor:
    """安全执行Python代码的类"""
    
    ALLOWED_MODULES = {
        'pandas': pd,
        'numpy': np,
        'os': os,
        'sys': sys,
        'datetime': None,
        'matplotlib': None,
        'matplotlib.pyplot': None,
        'seaborn': None,
        'scipy': None,
        'scipy.stats': None,
        'sklearn': None,
        'sklearn.cluster': None,
        'sklearn.decomposition': None,
        'sklearn.manifold': None,
    }
    
    DANGEROUS_PATTERNS = [
        'os.system', 'subprocess', 'eval', 'exec',
        '__import__', 'rm ', 'del ', 'drop ',
        'os.remove', 'shutil.rmtree', 'os.rmdir',
        'getattr', 'setattr', 'globals',
        'locals', 'vars', 'open(',
    ]
    
    def __init__(self, data: Optional[pd.DataFrame] = None, output_dir: str = 'output'):
        self.data = data
        self.output_dir = output_dir
        self.execution_history = []
    
    def check_safety(self, code: str) -> tuple[bool, Optional[str]]:
        """检查代码安全性"""
        import re
        
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in code.lower():
                return False, f"Dangerous pattern detected: {pattern}"
        
        if 'import' in code:
            import_pattern = r'^\s*import\s+(\w+)'
            from_pattern = r'^\s*from\s+(\w+)'
            
            for line in code.split('\n'):
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1)
                    if module not in self.ALLOWED_MODULES:
                        return False, f"Module '{module}' is not allowed. Allowed: {list(self.ALLOWED_MODULES.keys())}"
                
                match = re.match(from_pattern, line)
                if match:
                    module = match.group(1)
                    if module not in self.ALLOWED_MODULES:
                        return False, f"Module '{module}' is not allowed. Allowed: {list(self.ALLOWED_MODULES.keys())}"
        
        return True, None
    
    def create_execution_context(self) -> Dict[str, Any]:
        """创建代码执行上下文"""
        context = {
            'pd': pd,
            'np': np,
            'df': self.data.copy() if self.data is not None else None,
            'output_dir': self.output_dir,
            'result': None,
            'print_outputs': []
        }
        
        return context
    
    def save_code(self, code: str, filename: str = None) -> str:
        """保存生成的代码到文件"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        codes_dir = Path(self.output_dir) / 'generated_codes'
        codes_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"code_{timestamp}.py"
        
        filepath = codes_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return str(filepath)
    
    def execute(self, code: str) -> Dict[str, Any]:
        """执行Python代码"""
        safe, error_msg = self.check_safety(code)
        if not safe:
            return {
                'success': False,
                'error': f"Code safety check failed: {error_msg}",
                'result': None,
                'output': ''
            }
        
        saved_code_path = self.save_code(code)
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        context = self.create_execution_context()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, context)
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            result = context.get('result')
            if result is None:
                result = {
                    'success': True,
                    'message': 'Code executed successfully',
                    'data_shape': context['df'].shape if context['df'] is not None else None
                }
            
            self.execution_history.append({
                'code': code,
                'result': result,
                'success': True,
                'saved_code_path': saved_code_path
            })
            
            return {
                'success': True,
                'result': result,
                'output': stdout_output,
                'error': stderr_output if stderr_output else None,
                'data': context.get('df'),
                'saved_code_path': saved_code_path
            }
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.execution_history.append({
                'code': code,
                'error': str(e),
                'success': False
            })
            
            return {
                'success': False,
                'error': f"Execution error: {str(e)}\n{error_traceback}",
                'result': None,
                'output': stdout_capture.getvalue()
            }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history


class CodeGenerator:
    """根据任务描述生成Python代码"""
    
    def __init__(self, model_client):
        self.model_client = model_client
    
    def generate(
        self,
        task_description: str,
        data_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成Python代码"""
        
        prompt = self._create_prompt(task_description, data_info)
        
        messages = [
            {"role": "system", "content": "You are a Python code generator for transcriptome data analysis. Generate clean, safe, and effective code."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.model_client.chat(messages)
        
        code = self._extract_code(response)
        
        return code
    
    def _create_prompt(self, task_description: str, data_info: Optional[Dict[str, Any]] = None) -> str:
        """创建代码生成的prompt"""
        
        prompt = f"""Generate Python code to perform the following task:
{task_description}

"""
        
        if data_info:
            prompt += f"""
Available Data Information:
- Shape: {data_info.get('shape', 'N/A')}
- Columns: {', '.join(data_info.get('columns', []))}
- Data types: {data_info.get('dtypes', {})}

"""
        
        prompt += """Requirements:
1. Use pandas (pd) and numpy (np) for data manipulation
2. Use 'df' as the DataFrame variable name for the loaded data
3. Store output directory path in 'output_dir' variable
4. Create 'output_dir' directory if it doesn't exist using: import os; os.makedirs(output_dir, exist_ok=True)
5. Save results to files in the 'output_dir' directory
6. Return a dictionary named 'result' with keys:
   - 'success': boolean indicating if the operation was successful
   - 'message': string describing what was done
   - Any additional data you want to return (lists, dicts, etc.)

Example code structure:
```python
import pandas as pd
import numpy as np
import os

# Your data analysis code here
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Perform analysis
# ...

result = {
    'success': True,
    'message': 'Analysis completed successfully',
    'data': {...}  # Any data to return
}
```

IMPORTANT:
- Only output the Python code block, no explanations
- Do NOT use dangerous operations (os.system, eval, exec, etc.)
- Do NOT import modules other than: pandas, numpy, matplotlib, seaborn, scipy, sklearn
- Make sure to save results to the output directory
- Always create the 'result' dictionary at the end
```python
"""
        
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        import re
        
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        code_pattern = r'```\n?(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()
