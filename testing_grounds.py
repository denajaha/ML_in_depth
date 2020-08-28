import requests

# csv_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv'
csv_url = 'http://zinc15.docking.org/substances/ZINC000000000007/'
req = requests.get(csv_url)
print(req.text)

# csv_file = open('downloaded.csv', 'wb')
#
# csv_file.write(req)
# csv_file.close()


data = {
  'page.format': 'properties'
}

response = requests.post('http://zinc.docking.org/substances/', data=data)
#print(response.text)
