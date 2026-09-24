from osgeo import ogr, osr
import os

# Define the input file, output file, target CRS, and output format
input_file = "example.shp"  # Provide an example input shapefile
output_file = "reprojected_output.geojson"  # Output file name (GeoJSON)
output_format = "GeoJSON"  # Define the output format
target_epsg = 4326  # Target CRS EPSG code (WGS84)

# Open the input vector file
driver = ogr.GetDriverByName("ESRI Shapefile")
data_source = driver.Open(input_file, 0)  # 0 means read-only mode
if data_source is None:
    raise Exception(f"Failed to open input file: {input_file}")

layer = data_source.GetLayer()  # Get the input file's layer
source_srs = layer.GetSpatialRef()  # Get the source CRS

# Create the target spatial reference system
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(target_epsg)

# Create a coordinate transformation object
coord_transform = osr.CoordinateTransformation(source_srs, target_srs)

# Get the output driver
output_driver = ogr.GetDriverByName(output_format)
if os.path.exists(output_file):
    output_driver.DeleteDataSource(output_file)  # Delete the file if it already exists

# Create the output data source
output_data_source = output_driver.CreateDataSource(output_file)
if output_data_source is None:
    raise Exception("Failed to create an output data source.")

# Create the output layer with the same geometry type as the input layer
output_layer = output_data_source.CreateLayer(
    layer.GetName(), target_srs, geom_type=layer.GetGeomType()
)

# Copy fields (attributes) from the input layer to the output layer
input_layer_defn = layer.GetLayerDefn()
for i in range(input_layer_defn.GetFieldCount()):
    field_defn = input_layer_defn.GetFieldDefn(i)
    output_layer.CreateField(field_defn)

# Add features to the output layer with reprojected geometries
output_layer_defn = output_layer.GetLayerDefn()
for feature in layer:
    # Create a new feature
    output_feature = ogr.Feature(output_layer_defn)
    # Copy attributes
    output_feature.SetFrom(feature)

    # Transform geometry to the target CRS
    geom = feature.GetGeometryRef()
    geom.Transform(coord_transform)

    # Set the transformed geometry to the new feature
    output_feature.SetGeometry(geom)

    # Add the new feature to the output layer
    output_layer.CreateFeature(output_feature)

    # Destroy the feature to manage memory
    output_feature = None

# Close the datasets
data_source = None
output_data_source = None

print(f"Reprojected file has been saved as: {output_file}")
