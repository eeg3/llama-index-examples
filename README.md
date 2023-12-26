# llama-index-examples

LlamaIndex docs: https://docs.llamaindex.ai/en/stable/


#### API Authentication
Code expects API keys to stored in `.env` files:
```
GOOGLE_AI_STUDIO_KEY=""
OPENAI_API_KEY=""
```
- Google AI: [Get an API key](https://ai.google.dev/tutorials/setup)
- OpenAI: [API keys](https://platform.openai.com/api-keys)

## react-agent-query-tools

Utilize the [ReAct agent](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine.html) with multiple tools (PDF Doc Reader, [Wikipedia Loader](https://llamahub.ai/l/wikipedia)) that can be used to answer questions.

## multi-source-single-index-chat

Add documents from multiple sources (PDF Doc Reader, [Wikipedia Loader](https://llamahub.ai/l/wikipedia)) into a single index that can be queried. 