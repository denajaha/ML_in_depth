import requests

csv_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv'
req = requests.get(csv_url)

csv_file = open('downloaded.csv', 'wb')

csv_file.write(req)
csv_file.close()