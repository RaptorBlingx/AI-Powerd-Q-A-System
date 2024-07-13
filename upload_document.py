import requests

url = 'http://127.0.0.1:5000/add_document'
file_path = 'documents/Creamobile1.pdf'

with open(file_path, 'rb') as file:
    response = requests.post(url, files={'file': file})

print(response.json())
