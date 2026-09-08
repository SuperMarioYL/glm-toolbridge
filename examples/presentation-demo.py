import json
from glm_toolbridge import normalize_response
raw={"id":"demo","object":"chat.completion","model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"Checking the city","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":{"city":"Beijing"}}}]},"finish_reason":"tool_calls"}]}
result=normalize_response(raw).as_openai_dict()
print(json.dumps(result['choices'][0]['message'],ensure_ascii=False,indent=2))
