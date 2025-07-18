from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


SECURITY_EXAMPLES = [
        {
            "input": "def add(a, b):\n    return a + b",
            "output": "APPROVE: This is a simple addition function with no security risks."
        },
        {
            "input": "def find_substring(main_string: str, sub_string: str) -> int:\n   # Use the find method to locate the start index of the substring\n  return main_string.find(sub_string)",
            "output": "APPROVE: This is a simple addition function with no security risks."

        },
        {
            "input": "def multiply(a, b):\n    return a * b",
            "output": "APPROVE: This is a simple multiplication function with no security risks."
        },
        {
            "input": "def power(a, b):\n    return a ** b",
            "output": "APPROVE: This is a simple power function with no security risks."
        },
        {
            "input": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            "output": "APPROVE: This is a pure computation with no security risks."
        },
        {
            "input": "import os\nos.system('rm -rf /')",
            "output": "REJECT: This code executes a dangerous system command."
        },
        {
            "input": "import socket\ns = socket.socket()\ns.connect(('example.com', 80))",
            "output": "REJECT: This code opens a network connection, which is a security risk."
        },
        {
            "input": "with open" + "('/etc/passwd') as f:\n    data = f.read()",
            "output": "REJECT: This code reads a sensitive system file."
        }
    ]

EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["input", "output"],
    template="Input:\n{input}\nOutput:\n{output}"
)

SECURITY_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=SECURITY_EXAMPLES,
    example_prompt=EXAMPLE_TEMPLATE,
    prefix="""You are a security expert reviewing Python code for execution safety. 
Analyze ONLY the code provided for these security risks:
- File system access (reading/writing sensitive files)
- Network operations (HTTP requests, socket connections)
- System commands (os.system, subprocess)
- Import of dangerous modules
- Infinite loops or resource exhaustion
- Code obfuscation or suspicious patterns

APPROVE pure algorithms, computations, and data structures that don't use dangerous operations. 
REJECT only if you see actual dangerous operations in the code itself.

Here are some examples:""",
    suffix="Now look at this code and decide whether to APPROVE or REJECT, and explain your decision:\n{code}\nOutput:",
    input_variables=["code"],
    example_separator="\n\n"
)


def build_few_shot_prompt(code: str, security_prompt_template: str=SECURITY_PROMPT_TEMPLATE) -> str:
    return security_prompt_template.format(code=code)

