from enum import Enum


class OllamaModelType(Enum):
    """定義可從 Ollama 取得的模型"""

    LLAMA3_8B = "llama3:8b"
    """meta-llama/llama3:8b
    
    https://ollama.com/library/llama3:8b
    """

    PHI3_MINI = "phi3:mini"
    """microsoft/phi3:mini (3.8B)

    https://ollama.com/library/phi3:mini
    """
