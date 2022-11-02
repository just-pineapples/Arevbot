import streamlit as st
import pandas as pd
import os, sys, re
import plotly.express as px

sys.path.append(os.getcwd())

from datetime import time
from PIL import Image
from datetime import datetime as dt
from API_Intergration import expected_modeling_tags, inv_outages, inv_tags, plants_meta, all_plants, dc_outages, plants_coeffs, poa_tags

st.set_page_config(layout="wide")


path = "Arevon_home.png"
sys.path.append(path)

image = Image.open(fp=path)
st.sidebar.image(image)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def get_poa_tags(plant, start, end):
    df = poa_tags(plant, start, end)
    # df_1 = df.filter("MET")
    return df

@st.cache
def get_expected_tags(plant, start, end):
    df = expected_modeling_tags(plant, start, end)
    return df

@st.cache
def get_meta(plant):
    df = plants_meta(plant)
    return df

@st.cache
def get_inv_tags(plant,start,end):
    df = inv_tags(plant, start,end)
    return df

@st.cache
def dc_loss(plant, start, end, ac_limit):
    df = dc_outages(plant, start, end, ac_limit)
    return df

options = st.sidebar.selectbox("What values are we looking for today?", {"Plants Metadata/Actual Vs Expected", 'Inverter Outage', 'DC Outages', 'Actual POA'})

metadata = pd.read_excel("Project_Metadata_tables.xlsx", sheet_name='Plants')
_plants = [p[1] for p in metadata.values]



if options == "Inverter Outage":
    
    st.header('Inverter Outages - Lost Revenue')
    
    with st.sidebar:
        

        plants = st.selectbox("Plants", _plants)
        
        start_date = st.date_input("Start Date", value=dt(2022,7,1))
        end_date = st.date_input('End Date', value=dt.now())
        start_button = st.button("Submit")
    
    if start_button:
        
        inv = get_inv_tags(plants, start_date, end_date)
        _inv = inv.filter(regex="INV")
        
        fig = px.line(_inv)
        fig.update_xaxes(rangeslider_visible=True, 
                         rangeselector=dict(buttons=list([
                                        dict(count=1, label="1D", step="day", stepmode="backward"),
                                        dict(count=5, label="5 Min", step="minute", stepmode="todate"),
                                        dict(step='all')])))
        fig.update_xaxes(rangeselector=dict(visible=True))
        st.plotly_chart(fig, use_container_width=True)
        
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
    

elif options == "Plants Metadata/Actual Vs Expected":
    
    with st.sidebar:
        plants = st.selectbox("Plants", _plants)
        meta = get_meta(plants)
        
        start_date = st.date_input("Start Date", value=dt(2022,7,1))
        end_date = st.date_input('End Date', value=dt.now())
        if not start_date and end_date:
            st.warning("Please select the date you'd like to use")
            st.stop()
        # df = expected_modeling_tags(plants, start_date, end_date)
        df = get_expected_tags(plants, start_date, end_date)
        poa_names = list(df.filter(regex="IRRADIANCE_POA"))
        poa_sensor = st.multiselect("Which POA Sensor would you like to use?", poa_names)
        st.success("Please press submit to run code")
        start_button = st.button("Submit")

    if start_button:
        
        st.table(meta)
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
        # df.loc[:,'Meter_Power'] = (df.filter(regex = 'AC_POWER').median(axis = 1)).clip(0)
        df.loc[:,'Meter_Power'] = (df.filter(regex = 'MTR01')).clip(0)
    
        df.loc[:,'Expected_Energy'] = (derate_factor*ac_loss*df["POA"]*(poa + (poa2*df["POA"]) + (df['T_AMB']*poa_tamb) + (df['WS']*poa_ws))).clip(0,int(clipping_setpoint))
        
        expected_model = df[['Meter_Power','Expected_Energy']]
            
        fig = px.line(expected_model, title="Actual Vs Expected kW")
        st.plotly_chart(fig, use_container_width=True)
        
        
        res = df.index.to_series().diff().dt.total_seconds().fillna(0)[1]
        
        _expected_model = df[['Meter_Power','Expected_Energy',"POA"]]
        daily_loss = _expected_model.resample('1D').sum()/(1000*(60*60/res))
        daily_loss.loc[:,"Ratio"] = daily_loss["Meter_Power"]/daily_loss["Expected_Energy"] 
        daily_loss.loc[:,"Daily_Loss"] = daily_loss["Expected_Energy"]-daily_loss["Meter_Power"] 
        
        
        
        st.table(daily_loss)

        csv = convert_df(daily_loss)
        
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name=f'{plants} Actual vs Expected: {start_date} - {end_date}.csv',
            mime='text/csv'
        )
        st.info(f"Expected Energy Total: {daily_loss['Expected_Energy'].sum().round(2)}\n\n Actual Energy: {daily_loss['Meter_Power'].sum().round(2)}\n\n Our Losses: {daily_loss['Daily_Loss'].sum().round(2)}")
    

elif options == "DC Outages":
    
    
    
    plants = st.sidebar.selectbox("Plants", _plants)
    start_date = st.sidebar.date_input("Start Date", value=dt(2022,10,1))
    end_date = st.sidebar.date_input('End Date', value=dt.now())
    ac_limit = st.sidebar.number_input(label="Clipping Limit (kW)", value=int())
    
    start_button = st.sidebar.button("Submit")
    if start_button:
    
        cbx_outages = dc_loss(plants, start_date, end_date, ac_limit)
    
        # cbx_outages = dc_outages(plants, start_date, end_date, ac_limit)
        cbx_index = list(cbx_outages.index)
        d_strings = list(cbx_outages.columns)
        
        fig = px.imshow(cbx_outages, labels=dict(x="Downstrings", y="Dates", color="String Count"), x=d_strings, y=cbx_index, height=900, width=1500, text_auto=True, aspect="auto")
        fig.update_xaxes(side="top")
        fig.update_layout(yaxis_nticks=len(cbx_index))
        st.plotly_chart(fig, use_container_width= True)

        st.dataframe(cbx_outages)
        
        csv = convert_df(cbx_outages)
        
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name=f'{plants} DC_Underperformance: {start_date} - {end_date}.csv',
            mime='text/csv'
        )
    else:
        st.info("Please select what dates you would like to use.")
        
elif options == "Actual POA":

    with st.sidebar:
        plants = st.selectbox("Plants", _plants)
        start_date = st.date_input("Start Date", value=dt(2022,10,1))
        end_date = st.date_input('End Date', value=dt.now())
        _poa = get_poa_tags(plants, start_date,end_date)
        poa_selection = st.multiselect("Which POA Sensor do you wish to take out?", default=list(_poa.columns), options=list(_poa.columns))
        start_button = st.button("Submit")
        
    if start_button:
        st.subheader("You have selected: {}".format(",".join(poa_selection)))
        
        
        with st.container():
            
            select = {poa: _poa[_poa.filter(regex="MET")==poa] for poa in poa_selection}
            
            fig = px.line(_poa, labels=dict(x="Timestamp", value="POA m/s^2"), y=select)
            st.plotly_chart(fig, use_container_width=True)
    
        
            res = _poa.index.to_series().diff().dt.total_seconds().fillna(0)[1]
                
            dfs = _poa.resample("1D").sum()/(1000*(60*60/res))
            st.dataframe(dfs, width=1500)
            
            csv = convert_df(dfs)
            
            st.download_button(
                label="Download Table as CSV",
                data=csv,
                file_name=f'{plants} Actual POA.csv',
                mime='text/csv'
            )
            _poa.loc[:, "Actual POA"] = _poa.filter(items=select).median(axis=1)
            sum_total = _poa['Actual POA'].sum()/(1000*(60*60/res))
            st.info(f"Total POA for the month: {sum_total.round(2)}")
        