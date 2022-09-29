import pandas as pd
import time
from inverter_loss_calc import INV_Loss

start = time.time()

'''Will loop through each plant through its keys then runs whole script, remember to change title of Excel to match plants names'''

plants = {1: "Ayrshire",
        2: "Bearford 2",
        3: "Bizzell",
        4: "Bo Biggs" ,
        5: "Boaz",
        6: "Cline",
        7: "Haywood",
        8: "Hood",
        9: "ISS 35",
        10: "ISS 59",
        11: "ISS 60",
        12: "Meadowlark",
        13: "Moore",
        14: "Nash 97 Solar 2",
        15: "Nickelson 2",
        16: "Siler City 2",
        17:"St. Pauls 2",
        18:"Trinity Solar",
        19:"ZV2"}

for keys in plants.values():
  path = f"data/monthly_data/July/{keys} Historical - July 2022.xlsx"
  results = INV_Loss(path, keys)
  results.plants_monthly_loss()
  end = time.time()

print(f"Total time = {end-start}")
