from vicuna_llm import VicunaLLM

llm = VicunaLLM(server_url="http://localhost:52529")

print(llm("Hello, how are you?",stop=["\n###"]))