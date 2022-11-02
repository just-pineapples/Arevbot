import pandas as pd

from API_Intergration import expected_vs_actual, plants_coeffs



df = pd.read_excel("Lat - Sept.xlsx")


plant_coef = plants_coeffs("Latitude") 


poa = plant_coef["POA"].values
poa2 = plant_coef["POA^2"].values
poa_tamb = plant_coef["POA*TAMB"].values
poa_ws = plant_coef["POA*WS"].values
derate_factor = plant_coef["Factor"].values
ac_loss = plant_coef["AC loss"].values
dc_capacity = plant_coef["DC_CAP(kW)"].values
clipping_setpoint = plant_coef["Clipping Set Points kW"].values

df1 = df.set_index("Timestamp")
df1.index = pd.to_datetime(df1.index)
df1 = df1[~df1.index.duplicated(keep='first')]


df1 = df1[:-1]
df1.loc[:,'POA'] = df1.filter(regex = 'IRRADIANCE_POA').median(axis = 1)
df1.loc[:,'WS'] = df1.filter(regex = 'WIND_SPEED').median(axis = 1)
df1.loc[:,'T_AMB'] = df1.filter(regex = 'T_AMB').median(axis = 1)
df1.loc[:,'Meter_Power'] = (df1.filter(regex = 'AC_POWER').median(axis = 1)).clip(0)

df1.loc[:,'Expected_Power_V1'] = (derate_factor*ac_loss*df1["POA"]*(poa + (poa2*df1['POA']) + (df1['T_AMB']*poa_tamb) + (df1['WS']*poa_ws))).clip(0,int(clipping_setpoint))


expected_model = df1[['Meter_Power','Expected_Power_V1']]
res = df1.index.to_series().diff().dt.total_seconds().fillna(0)[1]

_expected_model = df1[['Meter_Power','Expected_Power_V1',"POA"]]
daily_loss = _expected_model.resample('1D').sum()/(1000*(60*60/res))
daily_loss.loc[:,"Ratio"] = daily_loss["Meter_Power"]/daily_loss["Expected_Power_V1"] 
daily_loss.loc[:,"Daily_Loss"] = daily_loss["Expected_Power_V1"]-daily_loss["Meter_Power"]

print(f"Expected Energy Total: {daily_loss['Expected_Power_V1'].sum().round(2)}\n\n Actual Energy: {daily_loss['Meter_Power'].sum().round(2)}\n\n Our Losses: {daily_loss['Daily_Loss'].sum().round(2)}")