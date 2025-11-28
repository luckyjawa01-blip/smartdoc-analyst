"""Code Execution Tool for SmartDoc Analyst.

This tool provides safe, sandboxed Python code execution
for calculations and data analysis tasks.
"""

import ast
import sys
from io import StringIO
from typing import Any, Dict, Optional
from .base_tool import BaseTool, ToolResult


class CodeExecutionTool(BaseTool):
    """Safe sandboxed Python code execution tool.
    
    Executes Python code in a restricted environment for
    performing calculations and data analysis. Includes
    safety checks to prevent dangerous operations.
    
    Attributes:
        timeout: Maximum execution time in seconds.
        max_output_size: Maximum output size in characters.
        allowed_modules: List of allowed imports.
        
    Example:
        >>> tool = CodeExecutionTool(timeout=30)
        >>> result = await tool.execute(
        ...     code="import math; result = math.sqrt(16)",
        ...     timeout=10
        ... )
    """
    
    # Safe built-in functions
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash',
        'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
        'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print',
        'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
        'str', 'sum', 'tuple', 'type', 'zip'
    }
    
    # Allowed modules for import
    ALLOWED_MODULES = {
        'math', 'statistics', 'datetime', 'json', 're',
        'collections', 'itertools', 'functools', 'operator'
    }
    
    # Dangerous patterns to block
    BLOCKED_PATTERNS = [
        'import os', 'import sys', 'import subprocess',
        '__import__', 'eval(', 'exec(', 'compile(',
        'open(', 'file(', 'input(',
        '__builtins__', '__globals__', '__locals__',
        'getattr', 'setattr', 'delattr',
        '.read(', '.write(', '.execute('
    ]
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 10000,
        allowed_modules: Optional[set] = None
    ):
        """Initialize the code execution tool.
        
        Args:
            timeout: Maximum execution time in seconds.
            max_output_size: Maximum output characters.
            allowed_modules: Set of allowed module imports.
        """
        super().__init__(
            name="code_execution",
            description="Execute Python code in a safe sandbox"
        )
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allowed_modules = allowed_modules or self.ALLOWED_MODULES.copy()
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute Python code safely.
        
        Args:
            code: Python code to execute.
            timeout: Override timeout setting.
            
        Returns:
            ToolResult: Execution result with output.
        """
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", self.timeout)
        
        if not code:
            return ToolResult(
                success=False,
                error="Code is required"
            )
            
        # Validate code safety
        safety_check = self._check_safety(code)
        if not safety_check["safe"]:
            return ToolResult(
                success=False,
                error=f"Code safety violation: {safety_check['reason']}"
            )
            
        try:
            result = self._execute_code(code, timeout)
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Execution error: {str(e)}"
            )
            
    def _check_safety(self, code: str) -> Dict[str, Any]:
        """Check code for safety violations.
        
        Args:
            code: Code to check.
            
        Returns:
            Dict: Safety check result.
        """
        # Check for blocked patterns
        code_lower = code.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return {
                    "safe": False,
                    "reason": f"Blocked pattern detected: {pattern}"
                }
                
        # Validate syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "safe": False,
                "reason": f"Syntax error: {str(e)}"
            }
            
        # Check AST for dangerous operations
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_modules:
                        return {
                            "safe": False,
                            "reason": f"Disallowed import: {alias.name}"
                        }
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.allowed_modules:
                    return {
                        "safe": False,
                        "reason": f"Disallowed import: {node.module}"
                    }
                    
            # Check for attribute access to dangerous names
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('__'):
                    return {
                        "safe": False,
                        "reason": f"Disallowed attribute access: {node.attr}"
                    }
                    
        return {"safe": True, "reason": None}
        
    def _execute_code(self, code: str, timeout: int) -> ToolResult:
        """Execute validated code.
        
        Args:
            code: Validated Python code.
            timeout: Execution timeout.
            
        Returns:
            ToolResult: Execution result.
        """
        # Prepare restricted globals
        restricted_globals = {
            '__builtins__': {
                name: getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name, None) or __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                for name in self.SAFE_BUILTINS
                if (hasattr(__builtins__, name) if not isinstance(__builtins__, dict) else name in __builtins__)
            }
        }
        
        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
                
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Execute code
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            
            # Get output
            output = sys.stdout.getvalue()
            
            # Get result variable if exists
            result_value = local_vars.get('result', local_vars.get('output', None))
            
            # Truncate output if too long
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\n... (truncated)"
                
            return ToolResult(
                success=True,
                data={
                    "output": output,
                    "result": result_value,
                    "variables": {
                        k: str(v)[:1000] for k, v in local_vars.items()
                        if not k.startswith('_')
                    }
                },
                metadata={
                    "timeout": timeout,
                    "output_length": len(output)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Runtime error: {str(e)}"
            )
        finally:
            sys.stdout = old_stdout
            
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds",
                    "default": self.timeout
                }
            },
            "required": ["code"]
        }
