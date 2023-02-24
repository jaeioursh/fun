import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import requests
import xml.etree.ElementTree as ET

lon='-97.0'
lat='38.0'
dist='1000.0'
res='100.0'
url='''https://graphical.weather.gov/xml/sample_products/browser_interface/ndfdBrowserClientByDay.php?whichClient=NDFDgenByDaySquare&lat=&lon=&listLatLon=&lat1=&lon1=&lat2=&lon2=&resolutionSub=&endPoint1Lat=&endPoint1Lon=&endPoint2Lat=&endPoint2Lon=&centerPointLat='''+lat+'''&centerPointLon='''+lon+'''&distanceLat='''+dist+'''&distanceLon='''+dist+'''&resolutionSquare='''+res+'''&zipCodeList=&citiesLevel=&format=12+hourly&startDate=2019-02-10&numDays=1&Unit=e&Submit=Submit'''

handle = requests.get(url)
text=handle.content
'''
with open("Output.txt", "w") as text_file:
    text_file.write(text)
print(text)
'''

root = ET.fromstring(text)
data={}
for point in root[1].findall('location'):
    name = point.find('location-key').text
    lat =  point.find('point').get('latitude')
    lon =  point.find('point').get('longitude')
    
    data[name]= [float(lat),float(lon)]
    
for point in root[1].findall('parameters'):
    name = point.get('applicable-location')
    value= point.find('probability-of-precipitation').find('value').text
    print value
    if value==None:
        value=0
    value=float(value)/100.0
    data[name].append(value)
    
dat=[]

for d in data:
    dat.append(data[d])
    
    
data=np.array(dat)
x,y,z=data.T[0],data.T[1],data.T[2]
map = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.scatter(x,y,c=z)
plt.show()

