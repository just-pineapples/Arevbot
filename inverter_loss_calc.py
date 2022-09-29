# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:20:33 2022

@author: brammohan
"""

'''Libraries needed'''

import glob
import os
from pprint import pprint as pp

import pandas as pd
import re
import cufflinks as cf
import itertools
import numpy as np

from datetime import datetime as dt

pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# tic = time.perf_counter()
cf.set_config_file(theme = 'pearl', sharing = 'public', offline = True)

''' Enter the path of your file '''
files = glob.glob("data\monthly_data\*.xlsx", recursive=True)

''' If the file is too long - read another xlsx file and concat/merge with the other file as well '''
#dff2 = pd.read_excel(path + '\\Moapa_May2022_2.xlsx')
#df = pd.concat([dff1,dff2])



class INV_Loss():

    def __init__(self, df, asset:str):

        plant = pd.read_excel(df)
        
        self.plants_name = asset
        self.df = plant

        self.inv_names = []
        self.rated_dc = []
        self.rated_ac = []

    def plant_metadata(self):
        
        df = self.df
        
        ''' Taking care of timestamps and indexing '''
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
  
        inv_names = self.inv_names
        
        ''' Loading inv power and poa into dataframe '''
        all_power = df.filter(regex = 'AC_POWER')
        inv_power = all_power.filter(regex = 'INV')
        all_power = all_power.fillna(0)
        

        all_power.loc[:,'POA'] = df.filter(regex = 'POA').mean(axis = 1) #Manual

    
        ''' Parsing inverter metadata from the inverter name '''


        inv_names = inv_power.columns.str.split('-').str[0].tolist()
        n = len(inv_power.columns.str.split('-').tolist()[0])
        
        rated_dc = self.rated_dc
        rated_dc = [re.findall(r"[-+]?(?:\d*\.\d+|\d+)",i) for i in inv_power.columns.str.split('-').str[n-2].tolist()]
        rated_dc = list(itertools.chain(*rated_dc))
        rated_dc = [float(i) for i in rated_dc]
        
        rated_ac = self.rated_ac
        rated_ac = [re.findall(r"[-+]?(?:\d*\.\d+|\d+)",i) for i in inv_power.columns.str.split('-').str[n-1].tolist()]
        rated_ac = list(itertools.chain(*rated_ac))
        rated_ac = [float(i) for i in rated_ac]
   
        ''' Creating metadata table '''
        metadata = pd.DataFrame({'Inv_name':inv_names,
                                'Rated_DC_KW':rated_dc,
                                'Rated_AC_KW':rated_ac})
   
        
        return metadata
    
    def inv_derates(self):
        '''To Determine Derates amoung Inverters'''
        
        metadata = self.plant_metadata()
        df = self.df
        
        df = df.set_index('Timestamp')
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        
        inv_names = self.inv_names
        
        ''' Loading inv power and poa into dataframe '''
        all_power = df.filter(regex = 'AC_POWER')
        inv_power = all_power.filter(regex = 'INV')
        all_power = all_power.fillna(0)

        all_power.loc[:,'POA'] = df.filter(regex = 'POA').mean(axis = 1) #Manual

        inv_names = inv_power.columns.str.split('-').str[0].tolist()

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
           
                
    def comms_outages(self):
        '''To Determine False Outages with Energy Delievered'''
    
    
    def inv_outages(self):
        
        metadata = self.plant_metadata()
        df = self.df
        
        df = df.set_index('Timestamp')
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        
        inv_names = self.inv_names
        
        ''' Loading inv power and poa into dataframe '''
        all_power = df.filter(regex = 'AC_POWER')
        inv_power = all_power.filter(regex = 'INV')
        all_power = all_power.fillna(0)

        all_power.loc[:,'POA'] = df.filter(regex = 'POA').mean(axis = 1) #Manual

        inv_names = inv_power.columns.str.split('-').str[0].tolist()

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
            inv1.iloc[:,i][(inv1.loc[:,'POA']<50)|(inv1.loc[:,'inv_model_power']<(0.01 * Inv_Limit))|(inv1.loc[:,'inv_model_power']>(0.99 * Inv_Limit))] = 0
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
        final_df['Project Name'] = self.plants_name
        final_df = final_df.reset_index(drop = True)
        final_df = final_df.set_index(["Project Name"])
        # print(final_df)
        if final_df.empty:
            print(f"No Inverters outages to report for {self.plants_name}")
        else:
            print("Well, we got some outages...") 
        return final_df

    def plants_ppa(self):
        '''Read through PPA Excel files and grab exact rates from plant'''
        
        ppa_rates = pd.read_excel('data/PPA_rates/Bala Export - 7.18.2022.xlsx')
        ppa_rates = ppa_rates.set_index('Project Name')
        ppa_rates = ppa_rates.loc[self.plants_name, ["Date", "$/MWh"]]
        ppa_rates['Date'] = pd.to_datetime(ppa_rates['Date'])
        ppa_rates['Date'] = ppa_rates['Date'].dt.strftime("%Y-%m")
        
        return ppa_rates 
    
    def plants_daily_loss(self):
        '''For Daily Plants performance'''
        pass        
    def plants_monthly_loss(self):
        
        '''Gather Both Dataframes from PPA rates and the Inverter outages'''
        energy_loss = self.inv_outages()
        energy_loss['Start_Date'] = pd.to_datetime(energy_loss['Start_Date'])
        energy_loss['End_Date'] = pd.to_datetime(energy_loss['End_Date'])
        df = energy_loss.loc[:, ['Start_Date','End_Date', 'Energy_loss_MWh']]
        plants_revenue = pd.DataFrame(self.plants_ppa())
        
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
        for re in result:
            energy_loss['Revenue Loss'] = re['$/MWh']*df.Energy_loss_MWh.values
            energy_loss['Revenue Loss'] = energy_loss['Revenue Loss'].round(2)
        """Saving Results to Results folder and matched with plants name""" 
        if not energy_loss.empty:
            print(f"Reporting for {self.plants_name}...")
            energy_loss.to_csv(f'data/Results/{self.plants_name} Results.csv')
        else:
            pass
            
        return df
    
    def plants_lifetime_loss():
        '''For Plants inverter outages longer than a month'''
                 

       

