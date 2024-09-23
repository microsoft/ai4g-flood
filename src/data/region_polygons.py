import geopandas as gpd  
from shapely.geometry import Polygon  

  
class CountryBoundaryManager:  
    def __init__(self, shapefile_path):  
        self.gdf = gpd.read_file(shapefile_path)  
  
    def get_country_polygon(self, country_name):  
        country = self.gdf[self.gdf['NAME_LONG'] == country_name]  
        if country.empty:  
            raise ValueError(f"No data found for country: {country_name}")  
        return country.geometry.values[0]  
  
    def add_custom_boundary(self, region_name, min_lat, max_lat, min_lon, max_lon):  
        # Create a Polygon from the bounding box coordinates  
        bounding_box_polygon = Polygon([  
            (min_lon, min_lat),  
            (min_lon, max_lat),  
            (max_lon, max_lat),  
            (max_lon, min_lat),  
            (min_lon, min_lat)  # Close the loop  
        ])  
          
        # Create a GeoDataFrame from the Polygon  
        custom_boundary_gdf = gpd.GeoDataFrame([{'geometry': bounding_box_polygon, 'NAME_LONG': region_name}], crs='EPSG:4326')  
          
        # Append the custom GeoDataFrame to the main GeoDataFrame  
        self.gdf = self.gdf.append(custom_boundary_gdf, ignore_index=True)  

boundary_manager = CountryBoundaryManager('./country_boundaries/ne_110m_admin_0_countries.shp')  
#example of adding a custom region for the globe
boundary_manager.add_custom_boundary('Global', min_lat=-70, max_lat=70, min_lon=-179.9, max_lon=179.9)  

def get_polygon(region):
    return boundary_manager.get_country_polygon(region)  
