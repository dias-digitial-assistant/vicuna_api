import dotenv
dotenv.load_dotenv()
from vicuna_llm import VicunaLLM

llm = VicunaLLM(server_url="http://localhost:52529")

print(llm("Hello"))