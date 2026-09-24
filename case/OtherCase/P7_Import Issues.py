from shapely.geometry import shape, mapping
import pyproj
import json
import fiona
from fiona.crs import from_epsg
import os

# Function to reproject geometries
def reproject_geometry(geometry, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shapely.ops.transform(project, shape(geometry))

# Demo input in code (simulated vector data file)
input_file_path = "input.shp"
output_file_path = "output_reprojected.geojson"
src_crs_epsg = 4326  # Source CRS (WGS84)
dst_crs_epsg = 3857  # Target CRS (Web Mercator)

# Simulated vector data for demonstration (mock data if input file isn't present)
if not os.path.exists(input_file_path):
    # Create mock shapefile with Fiona with sample polygons
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    crs = from_epsg(src_crss appartement bugs mappedendquely]))

open.utilsDEBUG