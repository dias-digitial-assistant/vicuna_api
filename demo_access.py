import requests
import json
from collections import deque
if __name__ == "__main__":

	# Get human input
	prompt = input()
	prompt = "### Human:"+prompt+"### Assistant:"
	headers = {"User-Agent": "fastchat Client"}
	pload = {
		"model": "vicuna-13b",
		"prompt": prompt,
		"max_new_tokens": 100,
		"temperature": float(0.5),
		"stop": "### Human:",
	}
	response = requests.post("http://localhost:52526" + "/worker_generate_stream", headers=headers,
			json=pload, stream=True)
	buffer = deque(maxlen=2)
	last_but_one_chunk = None

	for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
		if chunk:
			buffer.append(chunk)

	# If the buffer is full, the last but one chunk is the first element
	if len(buffer) == 2:
		last_but_one_chunk = buffer[0]

	# Process the last but one chunk
	if last_but_one_chunk:
		data = json.loads(last_but_one_chunk.decode("utf-8"))
		skip_echo_len = len(prompt.replace("</s>", " ")) + 1
		output = data["text"][skip_echo_len:].strip()
		print(f"Assistant: {output}")

	print("")
