"""
1. Save all images for each powerplant by year into a folder.
2. Save all segmentation images for each powerplant by year into a folder.
3. Get areas of segmentation smoke plumes, store in spreadsheets.
4. Look at each spreadsheet and create a row of data for master spreadsheet(sum all areas collumn, average area collumn)
5. Use these rows to create master spreadsheet.
    Collumn: num images, sum area, avg. area, id, location, ...
"""
import datetime
import os
from PIL import Image
import tifffile
import pandas as pd
import numpy as np

from get_images_utils import get_available_timestamps, get_images, segment_smoke_plumes, get_area_of_smoke_plumes

def create_one_location_dataset(data_path, year, coords, facility_id, longitude, latitude):
    if os.path.isdir(data_path) == False:
        os.mkdir(data_path)

    year_path = data_path + "/" + str(year) + "/"
    if (os.path.isdir(year_path) == False):
        os.mkdir(year_path)


    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year, 12, 31)
    #dates are inclusive

    #coords info
    #sample facility latitude, longitude: 33.63, -87.06
    #good distance apart: latitude: +-0.02, longitude: +-0.08
    #bounding box using longitude and latitude of lower left and upper right coordinates(try: (46.16, -16.15, 46.51, -15.58))


    folder = year_path + str(facility_id) + "/"

    collumns = ["Timestamp", "image_path", "seg_path", "contour_path", "smoke_plume_area"]
    df = pd.DataFrame(columns=collumns)
    print(df)

    timestamps = get_available_timestamps(start, end, coords, folder)
    total_smoke_plume_area = 0
    areas_one_year = []


    coords = (longitude-0.02, latitude-0.015, longitude+0.02, latitude+0.015)

    #coords = (longitude-0.025, latitude-0.02, longitude+0.025, latitude+0.02)

    for i, timestamp in enumerate(timestamps):
        image_path = get_images(timestamp, coords, folder)

        print(image_path)

        seg_path = segment_smoke_plumes(timestamp, image_path, folder)

        print(seg_path)

        contour_path, smoke_plume_area = get_area_of_smoke_plumes(timestamp, seg_path, folder)
        areas_one_year.append(smoke_plume_area)

        print(smoke_plume_area)
        total_smoke_plume_area += smoke_plume_area

        df.loc[len(df.index)] = [timestamp, image_path, seg_path, contour_path, smoke_plume_area]
        print(df)
    
    standard_deviation_smoke_plume_area = np.std(areas_one_year)

    df.to_csv(folder + facility_id + "_" + str(year) + ".csv")

    print("total smoke plume area: " + str(total_smoke_plume_area))

    average_smoke_plume_area = total_smoke_plume_area/len(timestamps)

    print("average smoke plume area: " + str(average_smoke_plume_area))

    return timestamps, total_smoke_plume_area, average_smoke_plume_area, standard_deviation_smoke_plume_area