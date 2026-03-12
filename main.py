from src import (
    DataLoader,
    DBEngine,
    Logger,
    DataPreparation,
    GraphConstructer,
    Visualizer,
    API,
)
import yaml
import sys


class SpatialContextEmbedder:
    def __init__(self, config_path: str = "config.yaml"):

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup logger
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path="logs/logs.log",
        )

        self.logger.info("Configuration loaded successfully. Starting Pipeline...")

        # Setup database connection
        db_config = self.config["database"]
        self.db_engine = DBEngine(
            dbuser=db_config["user"],
            dbpassword=db_config["password"],
            dbhost=db_config["host"],
            dbport=str(db_config["port"]),
            dbname=db_config["dbname"],
        )
        
        self.config_name = self.config.get("name", "default")

        # Setup data loader
        location_config = self.config["processing"]["location"]
        osm_config = self.config["processing"]["osm"]
        
        self.enable_private_score = self.config["processing"].get(
            "enable_private_score", False
        )
        self.private_cap_threshold = self.config["processing"].get("private_cap_threshold", 0.7)
        # mode: "api" # api or map
        self.mode = self.config.get("mode", "map")
        if self.mode == "api":
            self.logger.info("Initializing in API mode.")
            # coordinates are first CLI argument
            self.coordinates = sys.argv[1] if len(sys.argv) > 1 else None
            if not self.coordinates:
                self.logger.error("No coordinates provided for API mode.")
                sys.exit(1)
        else:
            self.coordinates = None

        self.data_loader = DataLoader(
            engine=self.db_engine.engine,
            enable_others_category=osm_config.get("enable_others_category", False),
            network_name=location_config["network_names"],
            city_name=location_config["city_name"],
            year=location_config["year"]
        )

    def run(self):
        """
        Main method to run the Spatial Context Embedding pipeline.
        """
        self.logger.info("Running Spatial Context Embedding Pipeline...")
        if self.mode == "api":
            self.logger.info("API Mode Initialization Phase Started")
            api = API(coordinates=self.coordinates, config_name=self.config_name)
            odlocations_df = api.data
        else:
            # Load coordinates and POIs
            self.logger.info("Graph Mode Initialization Phase Started")
            odlocations_df = self.data_loader.load_odlocations()

        prep_config = self.config["processing"]["data_preparation"]
        self.data_preparation = DataPreparation(
            max_air_distance_km=prep_config["max_air_distance_km"],
            max_walking_distance=prep_config["max_walking_distance"],
            osrm_endpoint=prep_config["osrm_endpoint"],
        )

        pois_df = self.data_loader.load_pois()
        if not self.enable_private_score:
            self.logger.info("Skipping Privacy Score as per Configuration")
        else:
            landuse_gdf = self.data_loader.load_osm_landuse()
            odlocations_df = self.data_preparation.odlocation_landuse(
                odlocations_df, landuse_gdf
            )

        pois_filtered_by_air = self.data_preparation.filter_locations_by_distance(
            odlocations_df, pois_df
        )

        pois_filtered_by_osrm = self.data_preparation.calculate_walking_distances_osrm(
            odlocations_df, pois_filtered_by_air
        )

        self.logger.info("Completed POI Walking Distance Calculations")

        self.logger.info("Spatial Context Embedding Process Completed")

        pois_opening_hours = self.data_preparation.parse_opening_hours_pois(
            pois_filtered_by_osrm
        )

        self.graph_constructer = GraphConstructer(
            odlocations_df,
            pois_filtered_by_osrm,
            pois_opening_hours,
            private_cap_threshold=self.private_cap_threshold,
        )

        self.graph_constructer.construct_graph()

        if self.mode == "api":
            results = api.process_coordinates(self.graph_constructer.get_graph())
            return results
        else:
            self.visualizer = Visualizer(self.graph_constructer.get_graph(), self.config_name)
            self.visualizer.plot_graph_map()


if __name__ == "__main__":
    embedder = SpatialContextEmbedder(config_path="config.yaml")
    embedder.run()
