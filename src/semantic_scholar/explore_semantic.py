import requests


r2 = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest').json()
print(r2)
[d['name'] for d in r2['datasets']]



r2

