from vicuna_llm import VicunaLLM

llm = VicunaLLM(server_url="http://localhost:52526")
# If you are using the container on the same server, replace localhost with the server IP address
llm.pload["temperature"] = 0.7
llm.pload["max_new_tokens"] = 1500

print(llm("Du bist der DIAS Assitant. Du bist sehr Nett und Hilfreich. Hallo, wie geht's?",stop=["\n### Human:"]))