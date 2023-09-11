from together_ai_api import TogetherLLM
import os
from dotenv import load_dotenv
load_dotenv()
llm = TogetherLLM(bearerToken=os.getenv("TOGETHER_BEARER_TOKEN"), model = "togethercomputer/llama-2-70b-chat")
# If you are using the container on the same server, replace localhost with the server IP address
llm.pload["temperature"] = 0.7
llm.pload["max_new_tokens"] = 1500

print(llm("Du bist der DIAS Assitant. Du bist sehr Nett und Hilfreich. Hallo, wie geht's?",stop=["\n### Human:"]))