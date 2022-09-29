
"""
Created on Tue Jul 19 10:32:54 2022

@author: brammohan
"""

import pandas as pd
import numpy as np
import os
import requests
import json
import datetime as dt

pd.set_option("display.max_rows", None)
''' Enter your path '''
# path = 'C:\\Users\\brammohan\\Documents\\Performance'
path = 'C:\\Users\dpinales\\Desktop\\Energy_Opt\\data\\Results'

''' Plant Name '''
# plant = "Boaz"
# start_date = dt.datetime(2022,6,1)
# end_date = dt.datetime(2022,6,30)

os.chdir(path)

# API credentials
subscription_key = "fa6da66b31e94018abac3eff8316027a"
customer_id = "2096"
base = 'https://api.powerfactorscorp.com/drive/v2'

# Functions for data download 

def get_all_plants(subscription_key, customer_key, base='https://api.powerfactorscorp.com/drive/v2', timeout=120):
    '''
    Params:
        subscription_key: users drive api subscription key, can be found in drive api portal
        customer_key: customer guid/key to be passed in to api request
        base: base url for drive api
        timeout: timeout value in seconds for api requests

    Returns:
        Dataframe of all plant ids and names belonging to customer
    '''

    endpoint = 'plant'
    url = '''{}/{}?&subscription-key={}'''.format(base, endpoint, subscription_key)
    r = requests.get(url, headers={'CustomerId': customer_key}, timeout=timeout)
    plants = pd.DataFrame.from_records(r.json())
    return plants


def get_plant(subscription_key, customer_key, p_id, base='https://api.powerfactorscorp.com/drive/v2', timeout=120):
    """
    Returns plant metadata for supplied plant id
    """

    endpoint = 'plant/' + p_id
    url = '''{}/{}?&subscription-key={}'''.format(base, endpoint, subscription_key)
    r = requests.get(url, headers={'CustomerId': customer_key}, timeout=timeout)
    return r.json()

def get_plants_metadata(s_key, c_id, p_id, base="https://api.powerfactorscorp.com/drive/v2", timeout=120):
    content_type = 'application/json'
    endpoint = "assets-metadata"
    url = "{}/{}?&subscription-key={}".format(base, endpoint, s_key)
    element_path = {'elementPaths':[p_id]}
    r = requests.post(url, json=element_path, headers={"Content-Type":content_type, "CustomerId": c_id}, timeout=timeout)
    meta = pd.DataFrame.from_records(r.json())
    
    meta_tags = [
             "Description",
             "Latitude",
             "Longitude", 
             "AC_Capacity", 
             "DC_Capacity", 
             "AC_CURTAILMENT_LIMIT",
             "Timezone"]
            #  "MOUNTING_GCR",
            #  "MOUNTING_TILT",
            #  "MOUNTING_TYPE",
            #  "MOUNTING_AZIMUTH

    new_tags = []

    for tags in meta['metadata']:
        for i in meta_tags:
            new_tags.append(tags[i])
        df = pd.DataFrame(new_tags, index=[meta_tags], columns=['Metadata'])
    
    return df
    
    
    
def get_plant_attributes(subscription_key, customer_key, plant_id, base='https://api.powerfactorscorp.com/drive/v2', timeout=120):
    '''
    Params:
        subscription_key: users drive api subscription key, can be found in drive api portal
        customer_key: customer guid/key to be passed in to api request
        base: base url for drive api
        timeout: timeout value in seconds for api requests

    Returns:
        Dataframe of all plant ids and names belonging to customer
    '''

    endpoint = 'plant/' + str(plant_id) + '/attribute'
    url = '''{}/{}?&subscription-key={}'''.format(base, endpoint, subscription_key)
    r = requests.get(url, headers={'CustomerId': customer_key}, timeout=timeout)
    plants = pd.DataFrame.from_records(r.json())
    return plants

def get_plant_devices(subscription_key, customer_key, plant_id, device_types=[], base='https://api.powerfactorscorp.com/drive/v2', timeout=120):
    """
    Returns queryable devices for the supplied plant id
        optionally filters results for specified device types
    """
    
    endpoint = 'plant/' + str(plant_id) + '/device'
    url = '''{}/{}?&subscription-key={}'''.format(base, endpoint, subscription_key)
    r = requests.get(url, headers={'CustomerId': customer_key}, timeout=timeout)
    plants = pd.DataFrame.from_records(r.json())
    
    if device_types:
        return plants[plants['type'].isin(device_types)]
    else:
        return plants

def get_device_attrs(subscription_key, customer_key, device_id, base='https://api.powerfactorscorp.com/drive/v2', timeout=120):
    """
    Returns queryable attributes for the supplied device id
    """
    
    
    endpoint = 'device/' + str(device_id) + '/attribute'
    url = '''{}/{}?&subscription-key={}'''.format(base, endpoint, subscription_key)
    r = requests.get(url, headers={'CustomerId': customer_key}, timeout=timeout)
    plants = pd.DataFrame.from_records(r.json())
    return plants


# Downloading large amount of data through parallel processing


import math
import logging
import concurrent.futures
from typing import Callable, Iterable
import arrow
import pytz



class TimeInterval():
    """
    Represents a time range and includes helper methods and fields for convenience
    """

    def __init__(self, *args, **kwargs):

        # Establish instance constants
        self.generate_interval(*args, **kwargs)

    def __str__(self):                                   
        return "%s - %s" % (self.starttime, self.endtime)

    def generate_interval(self, days_offset=1, starttime=None, endtime=None):

        if starttime and endtime:
            if endtime <= starttime:
                raise ValueError("End time must be greater than starttime")
            self.endtime = endtime
            self.starttime = starttime
        else:
            self.endtime = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            self.starttime = self.endtime - dt.timedelta(days=days_offset)

        self._intervallen = self.endtime - self.starttime

        # Find number of devices queryable per request based on Drive API restriction of asset count * days <= 7
        self._chunksize = max(1, math.floor((7 / self._intervallen.total_seconds()) * 24 * 60 * 60))
        
class Dispatch():
    
    def __init__(self, *args, **kwargs):

        self.initialize_globals(*args, **kwargs)
        self.initialize_default_settings()
    
    def initialize_globals(self, pf_key, pf_customer_id, *args, **kwargs):

        self.PF_API_MANAGER = PowerFactorsAPIManager(pf_key, pf_customer_id)

        # Generate time interval
        self.QUERY_INTERVAL = TimeInterval()

        self.LAST_SITE_REQUESTED = ""
    
    def initialize_default_settings(self):
        
        settings = {
            "update_devices": False,
            "recover": True,
            "replace": False,
            "dg": '',
            "mongo_insert_type": "documents"
        }
        
        self.SETTINGS = settings
    
    def fetch_data(self, device_ids: list, tags: list, on_finish: Callable=None, conserve_memory: bool=False, query_range: TimeInterval=None):
        """
        Queries Power Factors API for data for all input devices
            Queries are batched by devices as well as time frame to traverse Power Factors API limitations
        Args:
            device_ids: list of device ids to send in requests
                        all devices should be of the same type
            tags: list of attribute tags corresponding to devices to send in requests
            query_range: TimeInterval object containing information on the query time range
        Returns:
            A generator object that generates a dataframe for each request sent
        """

        if not query_range:
            query_range = self.QUERY_INTERVAL

        def send_and_format_pf_request(device_ids, tags, start_time=None, end_time=None, query_range=None, resolution='raw'):

            # Establish start and end query times
            if query_range:
                start_time = query_range.starttime
                end_time = query_range.endtime
            elif start_time and end_time:
                pass
            else:
                raise ValueError("Must specify query_range or starttime and endtime")
            
            # Request data
            data = self.PF_API_MANAGER.send_and_format_request(start_time, end_time, resolution, tags, device_ids, reqlimit=30)

            # Remove last data point as the end_time boundary is exclusive
            if not data.empty:
                data = data[:-1]

            return data

        # Fetch and format data
        # If query interval is longer than Drive API limitation, chunk requests by time period as well
        def iter_time_in_chunks(device_ids, tags, query_range):

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor_time:
                        
                cyclestart = query_range.starttime
                littledf = []

                while cyclestart < query_range.endtime:
                    cycleend = min(query_range.endtime, cyclestart + dt.timedelta(days=self.PF_API_MANAGER.POWER_FACTORS_QUERY_DURATION_LIMIT))

                    data = executor_time.submit(send_and_format_pf_request, device_ids=device_ids, tags=tags, start_time=cyclestart, end_time=cycleend)
                    littledf.append(data)

                    cyclestart = cycleend

                mediumdf = pd.concat(map(lambda x: x.result(), littledf))

            return mediumdf

        # Determine the function to call on each iteration of the device ids list
        # This is determined by the time period being requested
        if query_range._intervallen > dt.timedelta(days=self.PF_API_MANAGER.POWER_FACTORS_QUERY_DURATION_LIMIT):
            exec_function = iter_time_in_chunks
        else:
            exec_function = send_and_format_pf_request

        bigdf = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor_device:

            # For each chunk of devices
            for ids_chunk in self._chunker(device_ids, query_range._chunksize):

                # Execute the appropriate function with chunk of device ids
                littledf = executor_device.submit(exec_function, device_ids=ids_chunk, tags=tags, query_range=query_range)
                

                    # Keep track of the Future object by adding to a list
                bigdf.append(littledf)

            # Obtain the result of each Future execution (a dataframe) and concat into single df
            bigdf = pd.concat(map(lambda x: x.result(), bigdf), axis=1)

        bigdf = bigdf.sort_index()

        if on_finish and not bigdf.empty:
            on_finish(bigdf)

        return bigdf
    
    def _chunker(self, seq: Iterable, size: int) -> list:
        """
        Splits an iterable into chunks
        """

        return [seq[i:i+size] for i in range(0, len(seq), size)]

class PowerFactorsAPIManager():
    """
    Manager class to handle interactions with Power Factors Drive API
    """

    def __init__(self, pf_key, pf_customer_id):
        self.POWER_FACTORS_SUBSCRIPTION_KEY = pf_key
        self.POWER_FACTORS_CUSTOMER_ID = pf_customer_id
        self.POWER_FACTORS_QUERY_DURATION_LIMIT = 7

    def send_and_format_request(self, *args, **kwargs):
        """
        Sends and formats a request for data
        Returns an empty dataframe if invalid response is received
        """

        response = self.send_getdata_request_to_drive(*args, **kwargs)
        if response.status_code != 200:
            logging.critical("Received invalid %s response from Power Factors Drive API with message: %s" % (response.status_code, response.text))
            if response.text.split('.')[0] == "Out of call volume quota":
                #print(response.text.split('.'))
                exit(1)
            exit(0)
            return pd.DataFrame()
        data = self.transform_drive_api_response_to_df(response)

        return data


    def send_getdata_request_to_drive(
                                        self, 
                                        start_time: dt.datetime, 
                                        end_time: dt.datetime, 
                                        resolution: str, 
                                        attributes: list, 
                                        devices: list, 
                                        reqlimit: int=10, 
                                        timeout: int=60
                                    ) -> requests.models.Response:

        """
        Sends a POST getdata request to Power Factors API
        Will retry until reqlimit is met or response is received
        Args:
            start_time: start time of data query
            end_time: end time of data query
            resolution: resolution of data
            attributes: list of data tag names
            devices: list of device ids
            reqlimit: number of retries to attempt on failed connection
            timeout: number of seconds to wait for a response
        Returns:
            HTTP response from API
        """
    
        # Create request content
        base="https://api.powerfactorscorp.com/drive/v2"
        headers = {'customerId': self.POWER_FACTORS_CUSTOMER_ID}
        body = {
            "startTime":start_time,
            "endTime":end_time,
            "resolution":resolution,
            "attributes":attributes,
            "ids":devices
        }

        logging.debug("Fetching data from %s to %s for: %s" % (start_time, end_time, devices))
        
        # URL to point request to
        url = base+'/data?&subscription-key='+self.POWER_FACTORS_SUBSCRIPTION_KEY
            
        # Sends request and retries up to reqlimit times
        reqcount = 0
        received_resp = False
        while not received_resp and reqcount <= reqlimit:
            try:
                resp = requests.post(url, data=body, headers=headers, timeout=timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logging.debug(e)
            else:
                received_resp = True
            finally:
                reqcount+=1
                
        # If the number of failed requests has exceeded the request count limit, return a dummy response object with 408 error
        if reqcount > reqlimit:
            logging.critical("Failed to contact Power Factors Drive API after %s attempts" % reqcount)
            dummy = requests.models.Response()
            dummy.response_code = 408
            return dummy
        
        return resp

    def get_plant_devices_from_drive(self, plant_id: str, timeout: int=60) -> pd.DataFrame:
        """
        Query Power Factors API for devices belonging to a plant
        Args:
            plant_id: Power Factors ID of plant to request devices for
            timeout: Number of seconds to wait for API response before time out
        Returns:
            devices: DataFrame of all specified device types associated with the specified plant
        Raises:
            requests.exceptions.RequestException: Error sending api request
        """

        base='https://api.powerfactorscorp.com/drive/v2'
        endpoint = f'plant/{plant_id}/device'
        url = f'{base}/{endpoint}?&subscription-key={self.POWER_FACTORS_SUBSCRIPTION_KEY}'
        
        r = requests.get(url, headers={'CustomerId': self.POWER_FACTORS_CUSTOMER_ID}, timeout=timeout)
        
        if r.status_code == 200:
            return pd.DataFrame.from_records(r.json())
        else:
            raise ValueError("Received %s response from Drive API" % r.status_code)

    def transform_drive_api_response_to_df(self, r: requests.models.Response, indexed: bool=True) -> pd.DataFrame:
        '''
        Transforms the Drive API response into a dataframe.  Additional options include indexing.
        Args:
            r: response from a Drive API request, typically from drive_api_post
            indexed: boolean, return the data frame using a multi-level index or leave it as flat multi-key value pairs.
                Default is to index
        Returns:
            df : a data frame containing the transformed data.  Note, timestamps are converted to UTC
        Raises:
            KeyError: raises an exception
        '''

        #
        def create_type_label_from_id(label):
            if len(label.split('.')) == 3:
                return 'SITE'  # Plant
            else:
                return label[-5:-2]

        frequencies = {'00:05:00': '5min', '00:10:00': '10min', '00:15:00': '15min', '00:30:00': '30min', '01:00:00': 'H', '1.00:00:00': 'D'}
        r_norm = pd.DataFrame.from_records(r.json()['assets'])

        # remove empty assets
        r_norm = r_norm[r_norm.astype(str)['attributes'] != '[]']
        r_norm = r_norm.loc[r_norm.id != '0']

        r_norm['site'] = r_norm['id'].str[:11]  # extract site for index
        r_norm['type'] = r_norm['id'].apply(create_type_label_from_id)
        
        if r_norm.empty:
            return r_norm
        
        df = []
        for i in range(len(r_norm)):
            asset_list = []
            for j in range(len(r_norm.iloc[i]['attributes'])):
                tmp = pd.DataFrame.from_records(r_norm.iloc[i]['attributes'][j])
                tmp['values'] = pd.to_numeric(tmp['values'], errors='coerce')
                tmp['site'] = r_norm.iloc[i]['site']
                tmp['type'] = r_norm.iloc[i]['type']
                tmp['id'] = r_norm.iloc[i]['id']
                start=arrow.get(r_norm.iloc[i]['startTime']).datetime.astimezone(pytz.utc)
                end=arrow.get(r_norm.iloc[i]['endTime']).datetime.astimezone(pytz.utc)
                tmp['Timestamp'] = pd.date_range(start,
                                                end,
                                                freq=frequencies[r_norm.iloc[i]['interval']])
                asset_list.append(tmp)

            df.append(pd.concat(asset_list))
        df = pd.concat(df)

        # multi-index the data frame (default is True)
        if indexed == True:
            df.set_index(['site', 'type', 'id', 'name', 'Timestamp'], inplace=True)
            df = df.unstack(level=[0, 1, 2, 3])
            df.columns = df.columns.droplevel(0)
            df.sort_index(inplace=True, axis=1)
        return df





