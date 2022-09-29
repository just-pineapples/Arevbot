import pandas as pd
import time as t
import re
import itertools, os, sys

sys.path.append(os.getcwd())

from PF_API_Shared import get_all_plants, get_plant_devices, Dispatch,TimeInterval, get_plants_metadata
from src.config import SUBSCRIPTION_KEY, CUSTOMER_ID, timezone_convert 


t_start = t.time()

def all_plants():
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)
    return plants['name']

def plants_meta(plant:str):
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    meta = get_plants_metadata(SUBSCRIPTION_KEY, CUSTOMER_ID, plant_id)

    return meta

def plant_tz(plant):
    df = plants_meta(plant)
    tz_converter = timezone_convert(df)
    return tz_converter

def ppa_rates(plant):    
    ppa_rates = pd.read_excel("Bala Export - 7.18.2022.xlsx")
    ppa_rates = ppa_rates.set_index('Project Name')
    ppa_rates = ppa_rates.loc[plant, ["Date", "$/MWh"]]
    ppa_rates['Date'] = pd.to_datetime(ppa_rates['Date'])
    ppa_rates['Date'] = ppa_rates['Date'].dt.strftime("%Y-%m")
    return ppa_rates 

def plants_coeffs(plant):
    coeffs = pd.read_excel("Coronal Co-eff.xlsx")
    plant_coeffs = coeffs[coeffs["Plants"] == plant]
    return plant_coeffs

def expected_modeling_tags(plant:str, start, end):
    
    resolution = 'raw'
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    dispatch = Dispatch(SUBSCRIPTION_KEY, '2096')
    devices = get_plant_devices(SUBSCRIPTION_KEY,CUSTOMER_ID,plant_id)

    devtags = [
                ("Meter",["AC_POWER"]),
                ("Sensor Pyranometer POA", ["IRRADIANCE_POA"]),
                ("Sensor Temperature T_AMB", ["T_AMB"]),
                ("Sensor Anemometer", ["WIND_SPEED"])
        ]

    req_tags = []

    for dtype, tags in devtags:
        dtype_devices = devices[devices["type"]==dtype]
        
        df = dispatch.fetch_data(
            device_ids=dtype_devices['id'],
            tags = tags,
            query_range=TimeInterval(starttime=start,endtime=end))
        req_tags.append(df)
        
    
    all_data = pd.concat(req_tags, axis = 1)
    df_final = all_data.copy(deep=True)
    df_final.columns = [f"{deviceid}.{tag}" for site_id, dtype, deviceid, tag in df_final.columns]
    
    return df_final

def power_tags(plant, start, end):
    dev_tags = [("Inverter", ["AC_POWER"]),
                ("Meter",["AC_POWER"])]
    #DC Current & Curtailment     
    rq_tags = []
    
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    dispatch = Dispatch(SUBSCRIPTION_KEY, '2096')
    devices = get_plant_devices(SUBSCRIPTION_KEY,CUSTOMER_ID,plant_id)
    
    inv_description = devices[devices['description'].str.contains('INV|Inverter|Inveter')]
    
    temp_inv = [x for x  in inv_description["description"]]

    for dtype, tags in dev_tags:
        dtype_devices = devices[devices['type']==dtype]
        
        df = dispatch.fetch_data(dtype_devices['id'], 
                                 tags, 
                                 query_range=TimeInterval(starttime=start,endtime=end))
        rq_tags.append(df)
    inv_data = pd.concat(rq_tags, axis=1)
    data = inv_data.copy(deep=True)
    ident = [id for id in data.columns]
    
    remove_list = ["Inverter Pad", "SUBARRAY", "Module", 'Subarray']
    new_inv = []

    for s in remove_list:
        new_inv = [i for i in temp_inv if s[0] not in i and s[1] not in i and s[2] not in i and s[3] not in i]
    
    list_compile = lambda a,b: a + '-' +b
    mapper = list(map(list_compile, ident,new_inv))
    
    for i in ident:
        if "INV" not in i:
            mapper.append(i)
            
    data.columns = mapper
    
    
    return data

def inv_tags(plant:str, start, end):
    dev_tags = [("Inverter", ["AC_POWER"]),
                 ("Sensor Pyranometer POA", ["IRRADIANCE_POA"]),
                ]
      
    rq_tags = []
    
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    dispatch = Dispatch(SUBSCRIPTION_KEY, '2096')
    devices = get_plant_devices(SUBSCRIPTION_KEY,CUSTOMER_ID,plant_id)
    
    inv_description = devices[devices['description'].str.contains('kW')]
    
    inv = [x for x  in inv_description["description"]]

    for dtype, tags in dev_tags:
        dtype_devices = devices[devices['type']==dtype]
        
        df = dispatch.fetch_data(dtype_devices['id'], 
                                 tags, 
                                 query_range=TimeInterval(starttime=start,endtime=end))
        rq_tags.append(df)
    inv_data = pd.concat(rq_tags, axis=1)
    data = inv_data.copy(deep=True)
 
    ident = [id for site,type,id,name in data.columns]
    
    data.columns = ident
    list_compile = lambda a,b: a + '-' + b

    for i in data.columns:
        if "INV" in i:
            mapper = (list(map(list_compile, ident,inv)))
    df = data.filter(regex="INV")
    df.columns = mapper
    df.loc[:,"POA"] = data.filter(regex="MET").mean(axis=1)
    
    
    return df

def cbx_tags(plant:str, start, end):
    dev_tags = [("Combiner", ["DC_CURRENT"]),
            ("Sensor Pyranometer POA", ["IRRADIANCE_POA"]),
                ("Meter",["AC_POWER"])]
  
    rq_tags = []

    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    dispatch = Dispatch(SUBSCRIPTION_KEY, '2096')
    devices = get_plant_devices(SUBSCRIPTION_KEY,CUSTOMER_ID,plant_id)

    for dtype, tags in dev_tags:
        dtype_devices = devices[devices['type']==dtype]
        
        df = dispatch.fetch_data(dtype_devices['id'], 
                                    tags, 
                                    query_range=TimeInterval(starttime=start,endtime=end))
        rq_tags.append(df)
    
    inv_data = pd.concat(rq_tags, axis=1)
    data = inv_data.copy(deep=True)
    ident = [id for site, type, id, name in data.columns]
    data.columns = ident
    data = data.set_index([data.index])
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]
    _tz = plant_tz(plant)
    data.index = data.index.tz_convert(_tz)
    
    return data

def poa_tags(plant:str, start, end):
    dev_tags = [("Sensor Pyranometer POA", ["IRRADIANCE_POA"])]
  
    rq_tags = []

    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    dispatch = Dispatch(SUBSCRIPTION_KEY, '2096')
    devices = get_plant_devices(SUBSCRIPTION_KEY,CUSTOMER_ID,plant_id)

    for dtype, tags in dev_tags:
        dtype_devices = devices[devices['type']==dtype]
        
        df = dispatch.fetch_data(dtype_devices['id'], 
                                    tags, 
                                    query_range=TimeInterval(starttime=start,endtime=end))
        rq_tags.append(df)
    
    inv_data = pd.concat(rq_tags, axis=1)
    data = inv_data.copy(deep=True)
    ident = [id for site, type, id, name in data.columns]
    data.columns = ident
    data = data.set_index([data.index])
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]
    _tz = plant_tz(plant)
    data.index = data.index.tz_convert(_tz)
    
    return data

def dc_outages(plant:str, start, end):
    """For Filtered data, the data is filtered by the following parameters
        a. POA > 100
        b. (Meter Power > 1.5 MW) & (Meter Power < 25 MW)
        c. Hour > 12 (For shading purposes)
        d. All CBX's < 0 = 0
    """
    plants = get_all_plants(SUBSCRIPTION_KEY, CUSTOMER_ID)  
    plant_id = plants["id"][plants["name"] == plant].values[0]
    meta = get_plants_metadata(SUBSCRIPTION_KEY, CUSTOMER_ID, plant_id)
    
    data = cbx_tags(plant, start, end)

    curtailment_limit = meta.iat[5,0]
    
    data.loc[:, "Meter_Power"] = data.filter(regex="MTR").median(axis=1)
    data.loc[:, "POA"] = data.filter(regex="MET").median(axis=1)

    new_df = data.between_time("7:00", "17:00")
    new_df = data[(data.loc[:,"POA"] > 100)&(data.loc[:,"Meter_Power"]<curtailment_limit*0.99)]
    new_df[new_df<=0.00] = 0
    day_df = new_df.resample("1D").sum()
    

    dc_meta = pd.read_excel("Project_Metadata_tables.xlsx", sheet_name="Combiner_table").filter(items= ['plant_name', 'string_count','DC Rating Power kW', 'Module IMP'], axis=1)
    day_df = day_df.filter(regex="CMB")

    dc_issues = dc_meta[dc_meta["plant_name"]==plant]  
    dc_rating = [i for i in dc_issues['DC Rating Power kW'].values]

    relative_dc = []
    dc_expected = []

    max_dc = max(dc_rating)

    for i in dc_rating:
        dc = (i/max_dc)
        relative_dc.append(dc)

    string_count = [s for s in dc_issues['string_count']]

    cmb_relative = day_df.div(relative_dc)

    cmb_relative.loc[:,'norm'] = cmb_relative.apply(pd.Series.nlargest, axis = 1, n = 5).median(axis = 1)
    t = cmb_relative.div(cmb_relative['norm'], axis = 0).clip(lower = 0, upper = 1)
    t = t.iloc[:,:-1]

    down_strings = (string_count*(1-t)).round() 

    return down_strings
    
    

def inv_outages(plant:str, start, end):
    ''' Taking care of timestamps and indexing '''
    df = inv_tags(plant, start, end)
    
    inv_names = []
    df = df.set_index([df.index])
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    _tz = plant_tz(plant)
    df.index = df.index.tz_convert(_tz)
    
    ''' Loading inv power and poa into dataframe '''

    inv_power = df.filter(regex = 'INV')
    all_power = df.fillna(0)

    all_power.loc[:,'POA'] = df.filter(regex = 'POA').mean(axis = 1) #Manual


    ''' Parsing inverter metadata from the inverter name '''

    inv_names = inv_power.columns.str.split('-').str[0].tolist()
    n = len(inv_power.columns.str.split('-').tolist()[0])

    rated_dc = []
    rated_dc = [re.findall(r"[-+]?(?:\d*\.\d+|\d+)",i) for i in inv_power.columns.str.split('-').str[n-2].tolist()]
    rated_dc = list(itertools.chain(*rated_dc))
    rated_dc = [float(i) for i in rated_dc]

    rated_ac = []
    rated_ac = [re.findall(r"[-+]?(?:\d*\.\d+|\d+)",i) for i in inv_power.columns.str.split('-').str[n-1].tolist()]
    rated_ac = list(itertools.chain(*rated_ac))
    rated_ac = [float(i) for i in rated_ac]
    # Add Inverter names not tags
    ''' Creating metadata table '''
    metadata = pd.DataFrame({'Inv_name':inv_names,
                            'Rated_DC_KW':rated_dc,
                            'Rated_AC_KW':rated_ac})
    # print(metadata)
    ''' Assuming max_power as the expected inverter power '''

    all_power.loc[:,'inv_model_power'] = inv_power.max(axis = 1)
    inv_power.loc[:,'inv_model_power'] = inv_power.max(axis = 1)
    inv_power.loc[:,'POA'] = all_power.loc[:,'POA']

    ''' Dataframe to store all lost energy events '''
    final_df = pd.DataFrame()

    ''' Finding the min data resolution '''
    # data resolution
    res = df.index.to_series().diff().dt.total_seconds().fillna(0)[1]

    ''' Looping through every inverter and look for underperformance events '''
    for i in range(len(inv_names)):
        norm = (metadata['Rated_DC_KW'].iloc[i]/metadata['Rated_DC_KW'].max())
        Inv_Limit = metadata['Rated_AC_KW'].iloc[i]    
        inv1 = all_power.loc[(all_power.iloc[:,i] < (0.01*metadata['Rated_AC_KW'].iloc[i]))]#&(all_power.loc[:,'POA']>0)&(all_power.loc[:,'Meter_Power']>(0.1 * POI_Limit))]

        if len(inv1) == 0:
            continue
        
        """select column which contains the inverter name and energy delivered"""
        inv1.iloc[:,i] = inv1['inv_model_power'] * norm
        inv1.iloc[:,i][(inv1.loc[:,'POA']<100)|(inv1.loc[:,'inv_model_power']<(0.01 * Inv_Limit))|(inv1.loc[:,'inv_model_power']>(0.99 * Inv_Limit))] = 0
        inv = inv1.iloc[:,i]
        
        a = inv.index.to_series().diff().dt.total_seconds().fillna(res)
        b = a[a!=res]
        temp_df = pd.DataFrame(columns = ["Start_Date","End_Date","Equipment_Name","Loss_Category"])
        temp_df.loc[0,'Start_Date'] = str(a.index[0])
        
        for j in range(len(b)):
            row_index = a.index.get_loc(b.index[j])
            t_1 = row_index - 1
            temp_df.loc[j,'End_Date'] = str(a.index[t_1])
            temp_df.loc[j+1,'Start_Date'] = str(a.index[row_index])
        temp_df.loc[len(b),'End_Date'] = str(a.index[-1])    
        
        for j in range(len(temp_df)):
            temp_df.loc[j,'Energy_loss_MWh'] = inv[temp_df.loc[j,'Start_Date']:temp_df.loc[j,'End_Date']].sum()/(1000*(60*60/res))
            
        temp_df['Equipment_Name'] = metadata['Inv_name'].iloc[i]

        final_df = pd.concat([final_df,temp_df])

    ''' Writing energy loss events to a csv file '''


    final_df['Loss_Category'] = "Inverter Outage"
    final_df = final_df[final_df.loc[:,'Energy_loss_MWh'] > 1]
    final_df['Project Name'] = plant
    final_df = final_df.reset_index(drop = True)
    final_df = final_df.set_index(["Project Name"])

    if final_df.empty:
        print(f"No Inverters outages to report for {plant}")
    else:
        print("Well, we got some outages...") 
    energy_loss = final_df
    energy_loss['Start_Date'] = pd.to_datetime(energy_loss['Start_Date'])
    energy_loss['End_Date'] = pd.to_datetime(energy_loss['End_Date'])
    df = energy_loss.loc[:, ['Start_Date','End_Date', 'Energy_loss_MWh']]
    plants_revenue = ppa_rates(plant)
    
    '''Masking PPA Dates to one that matches the dates of the inverter outages'''
    mask = []
    result = []

    date = [e for e in plants_revenue.Date]

    for i in date:
        if df.Start_Date.where(cond=df['Start_Date'].dt.strftime("%Y-%m") == i).any():
            mask.append(i)
    for d, x in plants_revenue.iterrows():
        for m in mask:
            if x.Date == m:
                result.append(x)
    '''Once matched, the PPA AVG will be looped through and multiplied to the Energy Losses values'''
    for r in result:
        energy_loss['Revenue Loss'] = r['$/MWh']*df.Energy_loss_MWh.values
        energy_loss['Revenue Loss'] = energy_loss['Revenue Loss'].round(2)
    """Saving Results to Results folder and matched with plants name""" 
    if not energy_loss.empty:
        print(f"Reporting for {plant}...")
    else:
        pass
    
    return energy_loss


def expected_vs_actual(plant,start,end):
    
    plant_coef = plants_coeffs(plant)
    
    poa = plant_coef["POA"].values
    poa2 = plant_coef["POA^2"].values
    poa_tamb = plant_coef["POA*TAMB"].values
    poa_ws = plant_coef["POA*WS"].values
    derate_factor = plant_coef["Factor"].values
    ac_loss = plant_coef["AC loss"].values
    dc_capacity = plant_coef["DC_CAP(kW)"].values
    clipping_setpoint = plant_coef["Clipping Set Points kW"].values
    
    df = expected_modeling_tags(plant,start,end)
    df = df.set_index([df.index])
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
   
    df = df[:-1]
    df.loc[:,'POA'] = df.filter(regex = 'IRRADIANCE_POA').median(axis = 1)
    df.loc[:,'WS'] = df.filter(regex = 'WIND_SPEED').median(axis = 1)
    df.loc[:,'T_AMB'] = df.filter(regex = 'T_AMB').median(axis = 1)
    df.loc[:,'Meter_Power'] = (df.filter(regex = 'AC_POWER').median(axis = 1)).clip(0)
    
    df.loc[:,'Expected_Power_V1'] = (derate_factor*ac_loss*df["POA"]*(poa + (poa2*df['POA']) + (df['T_AMB']*poa_tamb) + (df['WS']*poa_ws))).clip(0,int(clipping_setpoint))
    
    expected_model = df[['Meter_Power','Expected_Power_V1']]
    
    return df

