from vicuna_llm import VicunaLLM

llm = VicunaLLM(server_url="http://localhost:52529")
llm.pload["temperature"] = 0.7
llm.pload["max_new_tokens"] = 1500

print(llm("Du bist der DIAS Assitant. Du bist sehr Nett und Hilfreich. Hallo, wie geht's?",stop=["\n### Human:"]))