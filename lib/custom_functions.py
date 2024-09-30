from BODSDataExtractor.extractor import TimetableExtractor
import pandas as pd
import numpy as np
import requests
from lxml import etree as et
# from xml.etree import ElementTree as et
from time import sleep
from datetime import datetime, timedelta
import xml.dom.minidom
import xmltodict
import json
from bs4 import BeautifulSoup as bsoup
import time
import schedule
import os
import shutil
import pytz
import re
import geopy.distance
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from pyproj import Geod
import uuid
import hashlib
import pandas as pd
import numpy as np
import glob
import os
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import matplotlib.dates as mdates
import folium
from folium import plugins
import h3
import geopy.distance
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from geojson import Feature, FeatureCollection
import geojson
import json
import matplotlib
import datetime
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import geopandas as gpd
import json
import requests
from shapely.geometry import Polygon, LineString, MultiLineString, Point
import shapely.ops as ops
from pyproj import Geod
from BODSDataExtractor.extractor import TimetableExtractor
import pandas as pd
import numpy as np
import requests
from lxml import etree as et
# from xml.etree import ElementTree as et
from time import sleep
from datetime import datetime, timedelta
import xml.dom.minidom
import xmltodict
import json
from bs4 import BeautifulSoup as bsoup
import time
import schedule
import os
import shutil
import pytz
import re
import uuid
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from folium import plugins
import h3
import geopy.distance

import plotly
import plotly.express as px
import plotly.figure_factory as ff

from geojson import Feature, Point, FeatureCollection
import geojson
import json
import matplotlib


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


# Variables
bristol_temple_meads = Point(-2.580549737314868, 51.449574343740345)
bristol_parkway = Point(-2.54327060775988, 51.514414056761495)
bristol_broadquay = Point(-2.5971874688706373, 51.45319747495111)
uwe_coor= Point(-2.5461254700104576, 51.49989442322204)
map_center = [51.497760, -2.567762]
bus_list = ['70', '72', '74', 'm1', 'm3', 'm4']
bus_speed_limit = 48

std_col_list = ['RecordedAtTime','ItemIdentifier','ValidUntilTime','LineRef','DirectionRef','DataFrameRef','DatedVehicleJourneyRef','PublishedLineName','OperatorRef','OriginRef','OriginName','DestinationRef','DestinationName','OriginAimedDepartureTime','DestinationAimedArrivalTime','Longitude','Latitude','LongitudeMatched','LatitudeMatched','BlockRef','VehicleRef','TicketMachineServiceCode','JourneyCode','VehicleUniqueId','TripId', 'EstimatedRoute']
unique_idetifier_cols = ['LineRef',  'DirectionRef', 'VehicleRef', 'BlockRef', 'DatedVehicleJourneyRef', 'OriginAimedDepartureTime', 'DestinationAimedArrivalTime']
siri_id_cols = ['LineRef', 'DirectionRef', 'DatedVehicleJourneyRef', 'BlockRef']
service_lookup_id_cols = ['LineName', 'Direction', 'JourneyCode', 'BlockNumber']
# import load_packages

# General Purpose Codes
def prettyprint(element, **kwargs):
    xml = et.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')

def df_insert_row(original_df, row_dictionary):
    # Convert row dictionary to DataFrame
    row_df = pd.DataFrame([row_dictionary])
    # Concatenate the original_df with the new row_df
    updated_df = pd.concat([original_df, row_df], ignore_index=True)
    return updated_df

def parse_runtime(duration):
    '''
    Translate DurationType PTxMyS into second values
    '''
    # Define the regular expression pattern
    pattern = re.compile(r'PT(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)
    
    if not match:
        raise ValueError(f"Invalid duration format: {duration}")
    
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = int(match.group(2)) if match.group(2) else 0
    
    total_seconds = minutes * 60 + seconds
    return total_seconds

def generate_uuid(*args):
    combined_string = ''.join(map(str,args))                              
    hash_value = hashlib.md5(combined_string.encode()).hexdigest()
    # print(hash_value)
    return str(uuid.UUID(str(hash_value)))

# XML File Processing

## For lxml.etree only
def remove_namespace(response_content):
    tree = et.fromstring(response_content)
    # Remove annoying namespaces 
    for elem in tree.iter():
        # print(elem.tag)
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # Strip namespace
        # Remove namespace from attributes
        attribs = elem.attrib
        # print(elem.attrib)
        for attrib in list(attribs.keys()):
            if '}' in attrib:
                new_attrib = attrib.split('}', 1)[1]
                attribs[new_attrib] = attribs.pop(attrib)
    et.cleanup_namespaces(tree)
    return et.tostring(tree)

# Data Extraction

def timetable_data_search(bus_list, api_key, org_name='', catalog_location='./docs/location_data_catalogue.csv'):
    location_data_catalogue = pd.read_csv(catalog_location)
    location_data_catalogue = location_data_catalogue[location_data_catalogue['Organisation Name'].str.contains(org_name)]
    search_list = list(location_data_catalogue['Datafeed ID'])
    feed_id_list = []
    for feed_id in search_list:
        response = requests.get("https://data.bus-data.dft.gov.uk/api/v1/datafeed/"+str(feed_id)+"/?api_key="+api_key)
        bus_siri_file = et.fromstring(response.content)
        buses = bus_siri_file.findall('.//{http://www.siri.org.uk/siri}LineRef')
        unique_bus = sorted(list(set(str.lower(b.text) for b in buses)))
        for bus in bus_list:
            if unique_bus.count(str.lower(bus)) > 0:
                feed_id_list.append((bus, feed_id))
        sleep(0.1)
    return feed_id_list

def location_data_search(bus_list, api_key, org_name='', catalog_location='./docs/location_data_catalogue.csv'):
    location_data_catalogue = pd.read_csv(catalog_location)
    location_data_catalogue = location_data_catalogue[location_data_catalogue['Organisation Name'].str.contains(org_name, case=False, na=False)]
    search_list = list(location_data_catalogue['Datafeed ID'])
    feed_id_list = []
    for feed_id in search_list:
        response = requests.get("https://data.bus-data.dft.gov.uk/api/v1/datafeed/"+str(feed_id)+"/?api_key="+api_key)
        bus_siri_file = et.fromstring(response.content)
        buses = bus_siri_file.findall('.//{http://www.siri.org.uk/siri}LineRef')
        unique_bus = sorted(list(set(str.lower(b.text) for b in buses)))
        for bus in bus_list:
            if unique_bus.count(str.lower(bus)) > 0:
                feed_id_list.append((bus, feed_id))
        sleep(0.1)
    return feed_id_list

def format_datetime_to_nearest_half_hour():
    now = datetime.now()
    # Round minutes to the nearest half hour
    if now.minute < 30:
        rounded_minute = 0
    else:
        rounded_minute = 30
    rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
    # Format to yymmddHHMM
    formatted_time = rounded_time.strftime('%H%M')
    return formatted_time

def write_data_to_csv(data, csv_path, backup_path = './data/raw/backup/', is_backup = True):
    data = pd.DataFrame(data)
    data = data[['RecordedAtTime','ItemIdentifier','ValidUntilTime','LineRef','DirectionRef','DataFrameRef','DatedVehicleJourneyRef','PublishedLineName','OperatorRef','OriginRef','OriginName','DestinationRef','DestinationName','OriginAimedDepartureTime','DestinationAimedArrivalTime','Longitude','Latitude','BlockRef','VehicleRef','TicketMachineServiceCode','JourneyCode','VehicleUniqueId']].copy()
    # Back up every 30 minutes
    backup_file = backup_path + str(os.path.basename(csv_path))[:-4] + '_' + str(format_datetime_to_nearest_half_hour()) + '.csv'
    if os.path.exists(csv_path):
        if is_backup:
            shutil.copy(csv_path, backup_file)
        data.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        data.to_csv(csv_path, mode='w', index=False, header=True)

# Geo-analytics Functions
def linestring_to_string(line):
    # Extract the coordinates from the LineString and format them
    coords = list(line.coords)
    coords_str = ','.join(f'({x},{y})' for x, y in coords)
    linestring_str = f'LineString([{coords_str}])'
    return linestring_str

def p2p_distance(line, point1, point2):
    '''
    enter a Linestring, and 2 Point shapely object
    returns the distance of two points on the line
    '''
    # Project the points onto the LineString to get distances along the LineString
    proj1 = line.project(point1, normalized=True)
    proj2 = line.project(point2, normalized=True)
    
    # Ensure proj1 is less than proj2 for correct segment extraction
    start_proj, end_proj = sorted([proj1, proj2])
    
    # line_gpd = gpd.GeoDataFrame(geometry=[line], crs=4326)
    geod = Geod(ellps="WGS84")
    line_length = geod.geometry_length(line)
    
    return (end_proj-start_proj)*line_length

def is_points_follow_line(line, point1, point2):
    '''
    enter a Linestring, and 2 Point shapely object
    returns True if point1 goes to point2 according to the line direction or point1=point2,
    returns False if point2 goes to point1 according to the line direction

    '''
    # Project the points onto the LineString to get distances along the LineString
    proj1 = line.project(point1, normalized=True)
    proj2 = line.project(point2, normalized=True)
    
    return True if proj1<=proj2 else False

def decode(encoded):
  '''
  Funtion provided by valhalla to decode location strings
  '''
  inv = 1.0 / 1e6
  decoded = []
  previous = [0,0]
  i = 0
  # For each byte
  while i < len(encoded):
    # For each coord (lat, lon)
    ll = [0,0]
    for j in [0, 1]:
      shift = 0
      byte = 0x20
      # Keep decoding bytes until you have this coord
      while byte >= 0x20:
        byte = ord(encoded[i]) - 63
        i += 1
        ll[j] |= (byte & 0x1f) << shift
        shift += 5
      # Get the final value adding the previous offset and remember it for the next
      ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
      previous[j] = ll[j]
    # Scale by the precision and chop off long coords also flip the positions so
    # Iits the far more standard lon,lat instead of lat,lon
    decoded.append([float('%.6f' % (ll[1] * inv)), float('%.6f' % (ll[0] * inv))])
  # Return the list of coordinates
  return decoded

def clutter_detection(data, w, r):
    '''
    data: a series containing 'lat' and 'lon' sorted in sequential order.
    r: the radius to identify the clutter. r can be estimate using the GPS error radius.

    return the index of the potential clutters
    '''
    core_list = data
    issue_list = []
    for i in range(1,len(core_list)):
        # print(i)
        start = max(0, i-int(w/2))
        stop = min(i+int(w/2), len(core_list)-1)
        if geopy.distance.geodesic(core_list[start], core_list[stop]).meters<r:
            issue_list.append(i)
    return issue_list

def clutter_grouping(data, w, s, record_count=150):
    '''
    data: a series of problematic points
    s: sensitivity determins how wide the gap is to merge two or more together into a bigger clutter
    w: the moving window size

    return a list of the start points and end points of the clutters
    '''
    if len(data)==0:
        return []

    clusters = []
    result = []
    current_group = [data[0]]

    for i in range(1, len(data)):
        # print(i)
        if data[i] - data[i-1] <= s:
            current_group.append(data[i])
        else:
            clusters.append(current_group)
            current_group = [data[i]]

    clusters.append(current_group)

    for j in clusters:
        # print(j)
        result.append([max(j[0]-int(w/2),0),min(j[-1]+int(w/2),record_count)])
    for k in range(0, len(result)-1):
        # print(k)
        if result[k][1]>result[k+1][0]:
            result[k][1] = int((result[k][1]+result[k+1][0])/2)
            result[k+1][0] = result[k][1]
    return result

def decluttering_data(data, w, r, s, lon_col_name='lon', lat_col_name='lat'):
    '''
    data: a Dataframe containing 'lat' and 'lon' sorted in sequential order.
    w: the moving window size
    r: the radius to identify the clutter. r can be estimate using the GPS error radius.
    s: sensitivity determins how wide the gap is to merge two or more together into a bigger clutter

    return the dataset clutters removed
    '''
    data = data.reset_index(drop=True)
    core_list = data[[lon_col_name,lat_col_name]].to_numpy()
    # print(core_list)
    issue_list = clutter_detection(core_list, w, r)
    # print(issue_list)
    clutter_list = clutter_grouping(issue_list, w, s, data.shape-1)
    # print(clutter_list)

    if len(clutter_list) > 0:
        # A list of uncluttered indeces
        index_list = []
        index = 0
        for i in range(0, len(clutter_list)):
            if clutter_list[i][0]<=0:
                clutter_list[i][0]=0
            if clutter_list[i][1]>=len(core_list)-1:
                clutter_list[i][1]=len(core_list)-1
            while index<=clutter_list[i][0]:
                index_list.append(index)
                # print(index_list)
                index+=1
            index=clutter_list[i][1]
            index_list.append(index)
            index+=1
            if i==len(clutter_list)-1:
                while index<=len(core_list)-1:
                    index_list.append(index)  
                    index+=1  
            
        # print(index_list)
        # result_df = data.iloc[index_list]

        return data.iloc[index_list]
    else:
        return data

def map_matching(data, lon_col_name='Longitude', lat_col_name='Latitude', time_col_name='RecordedAtTime',
                 search_radius=5, costing='bus', turn_penalty_factor=100):
    '''
    data: a dataframe containing lon, lat and time (seconds)

    return two results:
        a linestring for estimated route
        a DataFrame contains matched data on the exact order of input data
    '''
    data = pd.DataFrame({
        'lon': data[lon_col_name],
        'lat': data[lat_col_name],
        'time': data[time_col_name].apply(lambda x: x.timestamp())
    }).reset_index(drop=True)

    meili_coordinates = data.to_json(orient='records')
    meili_head = '{"shape":'
    meili_tail = f""","search_radius": {search_radius}, "shape_match":"map_snap", "costing":"{costing}", "format":"osrm", "turn_penalty_factor":{turn_penalty_factor}"""+"}"
    meili_request_body = meili_head + meili_coordinates + meili_tail
    # url = "http://localhost:8002/trace_route"
    url = "http://localhost:8002/trace_attributes"
    headers = {'Content-type': 'application/json'}
    request = requests.post(url, data=str(meili_request_body), headers=headers)

        # READ & FORMAT VALHALLA RESPONSE
    if request.status_code == 200:
        response_text = json.loads(request.text)
    else:
        print(f'API Error, {request.status_code}')
        return None, pd.DataFrame()

    matching = dict(response_text).get('shape')

    # lst_MapMatchingRoute = [LineString(decode(matching))]
    # estimated_route = gpd.GeoDataFrame(geometry=lst_MapMatchingRoute, crs=4326)
    lst_MapMatchingRoute = LineString(decode(matching))

    tracepoints = list(response_text.get('matched_points'))
    df_mapmatchedGPS_points = pd.DataFrame([(p['lon'],p['lat']) if (p is not None) else [None,None] for p in tracepoints] , columns=['LongitudeMatched', 'LatitudeMatched']).reset_index(drop=True)

    # result = pd.concat([data, df_mapmatchedGPS_points], axis=1)

    # df_mapmatchedGPS_points = df_mapmatchedGPS_points.loc[
    #     (df_mapmatchedGPS_points['lat'].notnull()) &
    #     (df_mapmatchedGPS_points['lon'].notnull())]
    
    return lst_MapMatchingRoute, df_mapmatchedGPS_points


def interpolate_line(line, num_points=100):
    distances = np.linspace(0, line.length, num_points)
    points = [line.interpolate(distance) for distance in distances]
    return LineString(points)

# Trip Processing

def classify_trip_trend(time_gaps, sensitivity=0.5):
    """
    Classifies the trip based on trend.
    The time gaps are split into three equal segments, and the trend of each segment is analyzed.
    Sensitivity is the threshold of standard deviation using to detect slowing down or speeding uo trends.
    Sensitivity = 0 means a disregard for std in comparing values --> if a - b > 0 then a is significantly larger than b
    Sensitivity = s means std is considered in comparing values --> if a - b > s*std then a is significantly larger than b
    Note that for Cosistently Ontime and Complex Trends still employs the std ithout modification
    """
    n = len(time_gaps)
    segment_size = n // 3
    
    if segment_size == 0:
        return np.nan

    # Split the time gaps into three segments
    segment1 = time_gaps[:segment_size]
    segment2 = time_gaps[segment_size:2*segment_size]
    segment3 = time_gaps[2*segment_size:]
    std_time = np.std(time_gaps)
    mean_time = np.mean(time_gaps)
    cv_time = std_time/mean_time

    # Calculate the mean time gap for each segment
    mean1 = np.mean(segment1)
    mean2 = np.mean(segment2)
    mean3 = np.mean(segment3)
    # d1 = mean1-mean_time
    # d2 = mean2-mean_time
    # d3 = mean3-mean_time
    
    # Classify based on trend
    if (abs(mean1 - mean_time) <= std_time) and (abs(mean2 - mean_time) <= std_time) and (abs(mean3 - mean_time) <= std_time):
        trend_class = "Consistent Trend"
    elif cv_time > 1:
        trend_class = "Complex Trend"
    elif mean2-mean1 > (std_time*sensitivity) and mean2-mean3 > (std_time*sensitivity):
        trend_class = "Mid-Trip Delay"
    elif mean1-mean2 > (std_time*sensitivity) and mean3-mean2 > (std_time*sensitivity):
        trend_class = "Mid-Trip Recovery"
    elif mean1 < mean3 and abs(mean3-mean1)>(std_time*sensitivity):
        trend_class = "Slowing Down"
    elif mean1 > mean3 and abs(mean3-mean1)>(std_time*sensitivity):
        trend_class = "Speeding Up"
    else:
        trend_class = "Complex Trend"
    
    return trend_class


# Series Aggregated Functions
def calculate_percentage(series):
    return (series.sum() / len(series)) * 100

def calculate_diversion_percentage(series):
    n=0
    for s in series:
        if s!=s:
            continue
        l = 0
        for i in s:
            l+=i['diverted_path_length']
        if l > 1500 and len(s) > 2:
            n+=1
    return (n/len(series)) * 100

def extract_diversion_data(series, field):
    result = []
    for s in series:
        e_result =[]
        if s!=s:
            continue
        for i in s:
            e_result.append(i[field])
        result.append(e_result)
    return pd.Series(result)