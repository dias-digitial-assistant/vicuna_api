from vicuna_llm import VicunaLLM

llm = VicunaLLM(server_url="http://localhost:52529")
llm.pload["temperature"] = 0.8
llm.pload["max_new_tokens"] = 1500

print(llm("Du bist sehr frohliche assistent, der Emojis mit dem Text antworten kann. Hallo, wie geht's?",stop=["\n### Human:"]))