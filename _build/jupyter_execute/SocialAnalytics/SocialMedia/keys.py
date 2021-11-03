gov_key = '9Vif2carl7eRwLscUNlio7MF7vKXPR9R9MwA3Bu9'
twit_key = 'zRK3Ag6JEBaVT4MxSOUpgaqIe'
twit_secr = '2h7R5e2jktfXc5u0HNJPJp5d7VTEVR8FImB89VCEeL1wpZLfZA'
twit_token = 'AAAAAAAAAAAAAAAAAAAAAP%2ByNgEAAAAALDNVyxMhq3kQ9aR7GWHx6t5tbVE%3DKGuE1bVwJdyecLmlwrRmH0ibomzKB9WSCyKEmJqa2K4P9upPxJ'

# import requests
# import json
# import prettytable
# headers = {'Content-type': 'application/json'}
# data = json.dumps({"seriesid": ['CUUR0000SA0','SUUR0000SA0'],"startyear":"2011", "endyear":"2014"})
# p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
# json_data = json.loads(p.text)
# for series in json_data['Results']['series']:
#     x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
#     seriesId = series['seriesID']
#     for item in series['data']:
#         year = item['year']
#         period = item['period']
#         value = item['value']
#         footnotes=""
#         for footnote in item['footnotes']:
#             if footnote:
#                 footnotes = footnotes + footnote['text'] + ','
#         if 'M01' <= period <= 'M12':
#             x.add_row([seriesId,year,period,value,footnotes[0:-1]])
#     output = open(seriesId + '.txt','w')
#     output.write (x.get_string())
#     output.close()

# data



# gov_key = 'cftiQ83bThdwbRQ2N8SejK1QGWqz8WJRza6n0fcz'#'DEMO_KEY'

# address_x = '1600 Amphitheatre Parkway, Mountain View, CA'
# api_key = gov_key
# url = 'https://developer.nrel.gov/api/utility_rates/v3.json'

# headers = {
#     'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
# }


# import requests

# headers = {
#     'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}

# url_1 = 'http://www.dsit.org.ir/?cmd=page&Cid=92&title=Kontakt&lang=fa'

# print(requests.get(url=url_1, headers=headers).text)

# S = requests.Session()

# #URL = "https://en.wikipedia.org/w/api.php"

# PARAMS = {
#     "api_key": api_key,
#     "address": address_x
# }

# try:
#     R = S.get(url=url, params=PARAMS, verify = False)
# except requests.exceptions.ConnectionError as e:
#     R = "No response"

# R

# R.text

# PARAMS = {
#     "api_key": api_key,
#     "address": address_
# }

# R = S.get(url=URL, params=PARAMS, verify=False)
# DATA = R.json()
# PAGES = DATA['query']['pages']

