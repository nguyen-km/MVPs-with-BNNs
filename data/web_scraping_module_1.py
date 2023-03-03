import requests
from bs4 import BeautifulSoup
import pandas as pd


url_start = "https://www.basketball-reference.com/awards/awards_{}.html"
years = list(range(1956,2021))
for year in years:
    url = url_start.format(year)
    data = requests.get(url)

    with open("mvp/{}.html".format(year), "w+") as f:
        f.write(data.text)

dfs = []
for year in years:
    with open("mvp/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
    soup = BeautifulSoup(page, "html")
    soup.find('tr', class_="over_header").decompose()
    if(1968 <= year <= 1976):
        read = 'all_nba_mvp'
    else:
        read = 'mvp'
    mvp_table = soup.find(id=read)
    mvp = pd.read_html(str(mvp_table))[0]
    mvp["year"] = year
    dfs.append(mvp)

mvps = pd.concat(dfs)
mvps.to_csv("mvp/clean/mvps.csv", index=False)