from urllib import request
import requests

files = {'file': open('test1.jpg', 'rb')}
headers = {'User-Agent': 'Chi dep trai bro'}
payload = {'files': open('test1.jpg', 'rb'),'seed':'123456'}
url = "http://127.0.0.1:8000/predict"
session = requests.Session()
# response = session.post(url, headers=headers, data=payload)   
response = session.post(url, headers=headers, files=files)
print("status", response.status_code)
# print("json", response.json())
print("Content:", response.content)

