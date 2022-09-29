import streamlit as st
import pandas as pd
import os, sys
import plotly.express as px

sys.path.append(os.getcwd())

from PIL import Image
from datetime import datetime as dt
from API_Intergration import expected_modeling_tags, inv_outages, plants_meta, all_plants, dc_outages, plants_coeffs, poa_tags
from src.config import get_project_root

st.set_page_config(layout="wide")


path = "Images\\Arevon_home.png"
root=get_project_root()

os.chdir(root)

image = Image.open(fp=path)
st.sidebar.image(image)

options = st.sidebar.selectbox("What values are we looking for today?", {"Plants Metadata/Actual Vs Expected", 'Inverter Outage', 'DC Outages', 'Monthly Reporting'})


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

if options == "Inverter Outage":
    
    st.header('Inverter Outages - Lost Revenue')
    plants = st.sidebar.selectbox("Plants", all_plants())
    
    start_date = st.sidebar.date_input("Start Date", value=dt(2022,7,1))
    end_date = st.sidebar.date_input('End Date', value=dt(2022,8,1))
    
    start_button = st.sidebar.button("Submit")
    
    if start_button:
        plants_inv = inv_outages(plants, start_date,end_date)
        if not plants_inv.empty:
            st.info(f"We have a total of {len(plants_inv['Energy_loss_MWh'])} Outages with Total of {plants_inv['Energy_loss_MWh'].sum().round(2)} MWh\n\n Total Revenue Lost {plants_inv['Revenue Loss'].sum().round(2)}$")
            st.dataframe(plants_inv)
        else:
            st.info("Looks like No Outages to report")
        csv = convert_df(plants_inv)
        
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name=f'{plants} inverter_outage.csv',
            mime='text/csv'
        )
    else:
        st.info("Please select what dates you would like to use.") 
    
    
    

if options == "Plants Metadata/Actual Vs Expected":
    plants = st.sidebar.selectbox("Plants", all_plants())
    meta = plants_meta(plants)
    st.table(meta)
    start_date = st.sidebar.date_input("Start Date", value=dt(2022,7,1))
    end_date = st.sidebar.date_input('End Date', value=dt(2022,8,1))
    
    
    
    df = expected_modeling_tags(plants, start_date, end_date)
 
    poa_names = list(df.filter(regex="IRRADIANCE_POA"))
    poa_sensor = st.sidebar.multiselect("Which POA Sensor would you like to use?", poa_names)
    start_button = st.sidebar.button("Submit")

    if start_button:
        
        plant_coef = plants_coeffs(plants)
    
        poa = plant_coef["POA"].values
        poa2 = plant_coef["POA^2"].values
        poa_tamb = plant_coef["POA*TAMB"].values
        poa_ws = plant_coef["POA*WS"].values
        derate_factor = plant_coef["Factor"].values
        ac_loss = plant_coef["AC loss"].values
        dc_capacity = plant_coef["DC_CAP(kW)"].values
        clipping_setpoint = plant_coef["Clipping Set Points kW"].values

        df = df.set_index([df.index])
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        df = df[:-1]
        
        df.loc[:, "POA"] = df.filter(items=poa_sensor,axis=1)
        df.loc[:,'WS'] = df.filter(regex = 'WIND_SPEED').median(axis = 1)
        df.loc[:,'T_AMB'] = df.filter(regex = 'T_AMB').median(axis = 1)
        df.loc[:,'Meter_Power'] = (df.filter(regex = 'AC_POWER').median(axis = 1)).clip(0)
    
        df.loc[:,'Expected_Power_V1'] = (derate_factor*ac_loss*df["POA"]*(poa + (poa2*df["POA"]) + (df['T_AMB']*poa_tamb) + (df['WS']*poa_ws))).clip(0,int(clipping_setpoint))
        
        expected_model = df[['Meter_Power','Expected_Power_V1']]
            
        fig = px.line(expected_model, title="Actual Vs Expected kW")
        st.plotly_chart(fig, use_container_width=True)
        
        
        res = df.index.to_series().diff().dt.total_seconds().fillna(0)[1]
        
        _expected_model = df[['Meter_Power','Expected_Power_V1',"POA"]]
        daily_loss = _expected_model.resample('1D').sum()/(1000*(60*60/res))
        daily_loss.loc[:,"Daily_Loss"] = daily_loss["Meter_Power"]/daily_loss["Expected_Power_V1"] 
        
        
        st.table(daily_loss)

        csv = convert_df(daily_loss)
        
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name=f'{plants} Actual vs Expected.csv',
            mime='text/csv'
        )
        st.info(f"Expected Total: {daily_loss['Expected_Power_V1'].sum().round(2)}\n\n Actual Power: {daily_loss['Meter_Power'].sum().round(2)}")
    
    else:
        st.info("Please select what dates you would like to use.")

if options == "DC Outages":
    
    
    
    plants = st.sidebar.selectbox("Plants", all_plants())
    start_date = st.sidebar.date_input("Start Date", value=dt(2022,7,1))
    end_date = st.sidebar.date_input('End Date', value=dt(2022,7,2))
    
    start_button = st.sidebar.button("Submit")
    if start_button:
    
    
        cbx_outages = dc_outages(plants, start_date, end_date)
        # sum_down_strings = sum(cbx_outages['Estimated_Downstrings'].values)
        # sum_string_count = sum(cbx_outages['String_count'].values)
        # ratio_performance = (sum_down_strings/sum_string_count)*100
        
        # st.info(f"The ratio of DC underperformance: {ratio_performance.round(2)}%")
        
                
        # string_count = cbx_outages['String_count'].values
        # down_str = cbx_outages['Estimated_Downstrings'].values
        
        st.dataframe(cbx_outages)
        csv = convert_df(cbx_outages)
        
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name=f'{plants} DC_Underperformance.csv',
            mime='text/csv'
        )
    else:
        st.info("Please select what dates you would like to use.")
        
if options == "Monthly Reporting":
    pass