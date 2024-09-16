import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from utils import plot_image

from sentinelhub import SHConfig

config = SHConfig()

#example power plant coords: longitude -87.06, latitude 33.63
coords = (-87.08, 33.61, -87.04, 33.65) #bounding box using longitude and latitude of lower left and upper right coordinates(try: (46.16, -16.15, 46.51, -15.58))
#coords = (-94.98-0.025, 39.45-0.02, -94.98+0.025, 39.45+0.02) #otherwise try +-0.015 for longtidue(larger nums usually) and +-0.01 for latitudes
coords = (-87.06-0.035, 33.65-0.03, -87.06+0.035, 33.65+0.03) #best one so far, zoomed out image means cloud blocking is good(big clouds, small plumes)
coords = (-87.06-0.025, 33.65-0.02, -87.06+0.025, 33.65+0.02)
coords = (-87.06-0.02, 33.63-0.015, -87.06+0.02, 33.63+0.015)
coords = (-94.98-0.015, 39.45-0.01, -94.98+0.015, 39.45+0.01)
coords = (-94.98-0.045, 39.45-0.04, -94.98+0.045, 39.45+0.04)
coords = (-83.35-0.035, 41.89-0.03, -83.35+0.035, 41.89+0.03)
coords = (-87.06-0.02, 33.63-0.015, -87.06+0.02, 33.63+0.015)

resolution = 15 #small is better
bbox = BBox(bbox=coords, crs=CRS.WGS84)
bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {bbox_size} pixels")



evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "CLP", "CLM"]
            }],
            output: {
                bands: 5
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02, sample.CLP, sample.CLM];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2018-03-01", "2021-04-09"), 
            #maxcc=0.01
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=bbox,
    size=bbox_size,
    config=config,
)


true_color_imgs = request_true_color.get_data()

print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")

image = true_color_imgs[0]
print(f"Image type: {image.dtype}")
print("mean: " + str(np.mean(image)))

print(true_color_imgs[1])
# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(image, factor=3.5 / 255, clip_range=(0, 1))