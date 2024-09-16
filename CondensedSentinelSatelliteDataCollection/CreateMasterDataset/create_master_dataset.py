"""
1. Save all images for each powerplant by year into a folder.
2. Save all segmentation images for each powerplant by year into a folder.
3. Get areas of segmentation smoke plumes, store in spreadsheets.
4. Look at each spreadsheet and create a row of data for master spreadsheet(sum all areas collumn, average area collumn)
5. Use these rows to create master spreadsheet.
    Collumn: num images, sum area, avg. area, id, location, ...
"""


"""
two locations activity:

1) 
"""
import datetime
import os
import numpy as np
from PIL import Image
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
import random
import matplotlib

from one_location_dataset_creator import create_one_location_dataset

df = pd.read_excel("/Users/26krisha/Desktop/SentinelSatelliteDataCollection/2021_data_summary_spreadsheets/ghgp_data_by_year.xlsx")
df.dropna(inplace=True) #in this dataset this removes all facilities that don't have data from every year(in the future make this just from 2016-2021 for Sentinel2)

#removes the text at the beginning and correctly assigns the collumns
df.columns = df.iloc[0]
df = df[1:]
df = df.reset_index(drop=True)

df = df.sort_values(by="2021 Total reported direct emissions", ascending=False)
df = df.head(100)
print(df.columns)

#take 2 random facilities
random_rows = []

while len(random_rows) < 1:
    random_num = (int)(random.random()*100) #generates a random number between 0(inclusive) and 100(exclusive)
    if random_num not in random_rows:
        random_rows.append(random_num)

#print(random_rows)
matplotlib.pyplot.close()

subset_df = df.iloc[random_rows]
#subset_df = df.loc[df['Facility Id'].isin([1007227])] #1001018, 1007227
#subset_df = df.loc[df['Facility Id'].isin([1000971])] 
subset_df = df.loc[df['Facility Id'].isin([1000971, 1000676])]
#print(subset_df.head())

#print(df.columns)
data_path_general = "image_data_v4"
data_path_synopsys_testing = "synopsys_proof_of_concept_v4"

columns = ['Location Id', 'Year', 'Number of Images', 'Total Area', 'Average Area', 'STD Area', 'Actual Emissions']
df_new = pd.DataFrame(columns=columns)

total_areas = []
average_areas = []
std_areas = []

for i in range(len(subset_df)):
    facility_id = str(subset_df.iloc[i]['Facility Id'])
    print(facility_id)
    latitude = subset_df.iloc[i]['Latitude']
    longitude = subset_df.iloc[i]['Longitude']

    #coords = (longitude-0.015, latitude-0.01, longitude+0.015, latitude+0.01)#v1
    coords = (longitude-0.025, latitude-0.02, longitude+0.025, latitude+0.02)
    print(coords)

    for index, j in enumerate([2018, 2019, 2020]): #iterates through the years 2018 to 2021 inclusive
        year = j

        timestamps, total_smoke_plume_area, average_smoke_plume_area, standard_deviation_smoke_plume_area = create_one_location_dataset(
                                                           data_path_general, year, coords, facility_id, longitude, latitude)
        
        print("facility id: " + str(facility_id))
        print("year: " + str(year))
        print("total area: " + str(total_smoke_plume_area))
        print("average area: " + str(average_smoke_plume_area))
        print("standard deviation area: " + str(standard_deviation_smoke_plume_area))
        
        #df_new.loc[len(df_new.index)] = [facility_id, year, len(timestamps), total_smoke_plume_area, average_smoke_plume_area, standard_deviation_smoke_plume_area]
        #df_new.to_csv("synopsys_proof_of_concept.csv")

        #read the csv files and create a dataset based off of that(facility_year)
        df_created = pd.read_csv(data_path_general + "/" + str(year) + "/" + str(facility_id) + "/" + str(facility_id) + "_" + str(year) + ".csv")
        #print(df_created.head(15))

        areas = (list)(df_created['smoke_plume_area'])
        #print(areas)
        total_smoke_plume_area = (int)(round(sum(areas), 0))
        total_areas.append(total_smoke_plume_area)

        average_smoke_plume_area = (int)(round(total_smoke_plume_area/len(areas), 0))
        average_areas.append(average_smoke_plume_area)

        standard_deviation_smoke_plume_area = (int)(round(np.std(areas), 0))
        std_areas.append(standard_deviation_smoke_plume_area)

        actual_emissions = (int)(round(subset_df.iloc[i][str(year) + " Total reported direct emissions"], 0))

        df_new.loc[len(df_new.index)] = [facility_id, year, len(df_created.index), total_smoke_plume_area, average_smoke_plume_area, standard_deviation_smoke_plume_area, actual_emissions]
        df_new.to_csv("synopsys_proof_of_concept_v6.csv")




data_path = "image_data"
year = 2021
coords = (-87.075, 33.62, -87.045, 33.64)
facility_id = 1007227

#total_smoke_plume_area, average_smoke_plume_area = create_one_location_dataset(
                                                    #data_path, year, coords,facility_id)