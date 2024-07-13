import requests

url = 'http://127.0.0.1:5000/query'
query = {
    "query": "What is the capital of France?"
}

response = requests.post(url, json=query)
print(response.json())
