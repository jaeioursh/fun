import json 
from urllib.request import urlopen



def load_champ(patch,champ):
    url="http://ddragon.leagueoflegends.com/cdn/"+patch+"/data/en_US/champion/"+champ+".json"
    f = urlopen(url)
    myfile = f.read()
    return json.loads(myfile)["data"][champ]

def load_items(patch):
    url="http://ddragon.leagueoflegends.com/cdn/"+patch+"/data/en_US/item.json"
    f = urlopen(url)
    myfile = f.read()
    return json.loads(myfile)

patch="11.24.1"
champ="Rengar"
data=load_champ(patch,champ)

#print(data)
for d in data["spells"]:
    print(d)