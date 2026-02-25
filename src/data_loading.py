from pathlib import Path
import pandas as pd
import geopandas as gpd
from sqlalchemy import text
from math import radians, sin, cos, sqrt, atan2

from .logger import Logger


class DataLoader:
    def __init__(
        self,
        engine,
        enable_others_category: bool = False,
        network_name=["N-MH", "N-MA"],
        city_name="Mannheim",
        year=2023,
    ):
        self.engine = engine
        self.enable_others_category = enable_others_category
        self.network_name = network_name
        self.city_name = city_name
        self.year = year
        self.entity_categories = {
            "food_beverage": ["restaurant", "fastfood", "cafe", "bakery", "bar"],
            "transportation": ["tram_stop", "station"],
            "education": ["school", "college", "university", "library"],
            "services_shopping": [
                "clothes_shop",
                "supermarket",
                "marketplace",
                "department_store_shop",
                "bank",
                "townhall",
                "place_of_worship",
                "dentist",
                "pharmacy",
                "chemist_shop",
            ],
            "entertainment": ["theatre", "cinema"],
        }

        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def load_odlocations(self) -> pd.DataFrame:
        """
        Loads all odlocations in network {self.network_name} with their important features
        """
        query = f"""
        SELECT 
            sl.station_id AS odlocation_id,
            sl.network_name,
            sl.location_id,
            sl.station_name as odlocation_name,
            ST_X(sl.location::geometry) as longitude,
            ST_Y(sl.location::geometry) as latitude,
            CASE 
                WHEN sl.last_seen::text LIKE '2025%' THEN 'aktiv'
                ELSE 'inaktiv'
            END as status
        FROM public.station_lifespans sl
        WHERE sl.location IS NOT NULL 
        AND sl.station_name not LIKE '%BIKE%'
        AND sl.station_name not LIKE '%recording%'
        AND sl.network_name in ({', '.join([f"'{net}'" for net in self.network_name])})
        ORDER BY sl.last_seen DESC;
        """

        with self.engine.connect() as conn:
            odlocations_df = pd.read_sql(text(query), conn)
        self.logger.info(
            f"Loaded O/D-Locations in Network {self.network_name}: {len(odlocations_df)}"
        )
        odlocations_df = odlocations_df.drop_duplicates(
            subset=["odlocation_id"], keep="first"
        )
        self.logger.info(
            f"After removing duplicates, {len(odlocations_df)} unique O/D-Locations remain."
        )

        return odlocations_df

    def load_pois(self) -> pd.DataFrame:
        """
        Loads all POIs with their important features
        POIs are stored in public.osm, location information in geo_information
        Only POIs in {self.city_name} ({self.network_name}) are considered
        """
        query = f"""
        SELECT 
            o.id as poi_id,
            o.location_id,
            o.name as poi_name,
            o.entity_name as entity_name,
            o.cuisine,
            ST_X(g.location::geometry) as longitude,
            ST_Y(g.location::geometry) as latitude,
            o.opening_hours,
            g.city_name as city,
            g.postal_code,
            g.federal_state_name as state,
            g.country_name as country
        FROM public.osm o
        JOIN public.geo_information g ON o.location_id = g.location_id
        WHERE g.location IS NOT NULL
        AND g.city_name = '{self.city_name}'
        """

        with self.engine.connect() as conn:
            pois_df = pd.read_sql(text(query), conn)
        self.logger.info(
            f"Loaded POIs in {self.city_name} ({self.network_name}): {len(pois_df)}"
        )
        pois_df = pois_df.dropna(subset=["poi_name"])
        pois_df = pois_df.drop_duplicates(
            subset=["poi_name", "entity_name"], keep="first"
        )

        self.logger.info(
            f"After removing duplicates, {len(pois_df)} unique POIs remain."
        )

        # add category column
        pois_df["category"] = "other"
        for category, entities in self.entity_categories.items():
            pois_df.loc[pois_df["entity_name"].isin(entities), "category"] = category

        if not self.enable_others_category:
            # remove 'other' category POIs
            pois_df = pois_df[pois_df["category"] != "other"]
            self.logger.info("Removed 'other' category POIs from the dataset.")

        # count frequency per category
        pois_counts = pois_df["category"].value_counts().reset_index()
        pois_counts.columns = ["category", "count"]

        return pois_df

    def load_osm_landuse(self) -> gpd.GeoDataFrame:
        """
        Loads OSM Landuse data from the database. Returns a GeoDataFrame with geometries.
        """
        sql = f"""
                SELECT
                    id,
                    city,
                    landuse,
                    ST_AsText(area) as wkt_geom
                FROM public.osm_landuse
                WHERE city = '{self.city_name}'
        """
        with self.engine.connect() as conn:
            landuse_df = pd.read_sql(text(sql), conn)
        landuse_df["geometry"] = gpd.GeoSeries.from_wkt(landuse_df["wkt_geom"])
        landuse_gdf = gpd.GeoDataFrame(landuse_df, geometry="geometry")
        self.logger.info(
            f"Loaded OSM Landuse data for {self.city_name}: {len(landuse_gdf)} records"
        )
        return landuse_gdf

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points on the Earth.
        """

        R = 6371.0  # Earth radius in kilometers

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance
