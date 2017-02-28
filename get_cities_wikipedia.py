from bs4 import BeautifulSoup
import pandas as pd
import urllib2
wiki = "https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Colorado"
header = {'User-Agent': 'Mozilla/5.0'} #Needed to prevent 403 error on Wikipedia
req = urllib2.Request(wiki,headers=header)
page = urllib2.urlopen(req)
soup = BeautifulSoup(page)

area = ""
district = ""
town = ""
county = ""
table = soup.find("table", { "class" : "wikitable sortable" })

co_cities_table = []
for row in table.findAll("tr"):
    cells = row.findAll("td")
    #For each "tr", assign each "td" to a variable.
    if len(cells) == 16:
        city = cells[0].find(text=True)
        latlng = cells[15].findAll(text=True)[-1].split('; ')
        city_lat, city_lng = float(latlng[0]), float(latlng[1])
        co_cities_table.append({'city':city, 'city_lat':city_lat, 'city_lng': city_lng})

cities_df = pd.DataFrame(co_cities_table)
cities_df.to_csv('colorado_cities.csv', header=True, encoding='utf-8')
if __name__ == '__main__':
    pass
