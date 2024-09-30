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
from lib.custom_functions import *


# api key
api = 'a3eda657579ef98d499ef515fbb5a32a86b22248'
fb_noc = ['FBRI']
timetable_dataset_id_list = [5815, 5814, 5813, 2283]
location_dataset_id_list = [5815, 5814, 5813, 2283]
bus_list = ['70', '72', '74', 'm1', 'm3', 'm4']
dataset_id = 699
latest_log = "Log initiating...\n"

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

def bus_loc_data_collection(dataset_id, api_key, bus_list):
    try:
        recording_df = pd.DataFrame()
        location_raw = requests.get("https://data.bus-data.dft.gov.uk/api/v1/datafeed/"+str(dataset_id)+"/?api_key="+api_key)
        location_data = et.fromstring(remove_namespace(location_raw.content))
        file_name = str(datetime.now().strftime('%Y%m%d'))+'_'+str(dataset_id)
        log_file_name = str(datetime.now().strftime('%Y%m%d'))+'_'+str(dataset_id)+'_log'
        log_message = ""
        for line in bus_list:
            for journey in location_data.xpath(".//VehicleActivity[MonitoredVehicleJourney/LineRef/text()='"+line+"']"):
                journey_info_dict = {}
                for attributes in journey.iter():
                    if attributes.text != None:
                        journey_info_dict[attributes.tag] = attributes.text
                        # print(f'{attributes.tag} - {attributes.text}')
                if 'DestinationAimedArrivalTime' in journey_info_dict:
                    # Only record when recorded time is less than bus deadline
                    # if datetime.strptime(journey_info_dict['DestinationAimedArrivalTime'], '%Y-%m-%dT%H:%M:%S%z') >= datetime.now(pytz.timezone('UTC')):
                    #     # print('ACTIVE LINES as of '+str(datetime.now(pytz.timezone('UTC'))))
                    #     # print(journey_info_dict)
                    #     recording_df = df_insert_row(recording_df, journey_info_dict)    
                    recording_df = df_insert_row(recording_df, journey_info_dict)    
        write_data_to_csv(recording_df, csv_path='./data/raw/'+file_name+'.csv')
        # print('--------------------')
        with open('./data/raw/'+log_file_name, "a+") as f:
            log_message = f"""Logging at {datetime.now()}\n\t\tTotal unique buses: {recording_df['VehicleRef'].nunique()}\n"""
            f.write(log_message)
            # f.write(f"Logging at {datetime.now()}\n")
            # f.write(f"\t\tNew data entry retrieved: {recording_df.shape[0]}\n")
            # f.write(f"\t\tTotal unique buses: {recording_df['VehicleRef'].nunique()}\n")
            # f.write(f"\t\tTotal active buses: {recording_df['VehicleRef'].nunique()}\n")
            f.flush()
    except Exception as error:
        # print("An exception occurred")
        with open('./data/raw/'+log_file_name, "a+") as f:
            log_message = f"""Logging at {datetime.now()}\n\t\tAn exception occurred: {type(error).__name__} – {error}\n"""
            f.write(log_message)
            # f.write(f"\t\tTotal active buses: {recording_df['VehicleRef'].nunique()}\n")
            f.flush()
    # print(log_message)
    with open('./data/raw/latest_log', "w+") as f:
            f.write(log_message)
            f.flush()


# Schedule the write function to run every 5 seconds
schedule.every(5).seconds.do(bus_loc_data_collection, dataset_id, api, bus_list)

# Run the scheduled tasks
if __name__ == '__main__':
    while True:
        schedule.run_pending()
        time.sleep(1)
        # if datetime.now().hour >= 18:
        #     break
