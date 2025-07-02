from asyncio import to_thread

class GeminiLangchainLLM:
    def __init__(self, langchain_llm):
        self.langchain_llm = langchain_llm

    def _get_prompt_str(self, prompt):
        # Handle LangChain prompt value types
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    async def agenerate_prompt(self, prompt, **kwargs) -> str:
        prompt_str = self._get_prompt_str(prompt)
        result = await to_thread(self.langchain_llm.invoke, prompt_str)
        return result.content if hasattr(result, 'content') else str(result)

    def generate_prompt(self, prompt, **kwargs) -> str:
        prompt_str = self._get_prompt_str(prompt)
        result = self.langchain_llm.invoke(prompt_str)
        return result.content if hasattr(result, 'content') else str(result)
