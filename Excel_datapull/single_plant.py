import pandas as pd
import time

from inverter_loss_calc import INV_Loss

start = time.time()
end = time.time()

'''For Single Plants, Type name and make sure Excel file is named similiar to plants name'''
plant = "ISS 59"

august = f"data/monthly_data/August/{plant} Historical - August 2022.xlsx"
# custom_date = f"data/custom_data/{plant} Historical - Last 7 days_ Aug 11, 2022 - Aug 17, 2022.xlsx"
results = INV_Loss(august, plant)
results.plants_monthly_loss()

print(f"Total time = {end-start}")




   


