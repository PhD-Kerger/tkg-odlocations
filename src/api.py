from pathlib import Path
from src import Logger
import pandas as pd
import networkx as nx
import json
import matplotlib.pyplot as plt


class API:
    def __init__(self, coordinates: list = None, config_name: str = "default"):
        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )
        # coordinates are in format 8.495,49.453,Station1;8.496,49.454,Station2;... as string
        # station_name is optional (third parameter after lat,lon)
        # make list of tuples by splitting on ';' and then on ','

        self.config_name = config_name
        self.coordinates = []
        if coordinates:
            if "+" not in coordinates:
                parts = coordinates.split(",")
                lat, lon = map(float, parts[:2])
                station_name = parts[2] if len(parts) > 2 else None
                self.coordinates = [(lat, lon, station_name)]
            else:
                for coord in coordinates.split("+"):
                    parts = coord.split(",")
                    lat, lon = map(float, parts[:2])
                    station_name = parts[2] if len(parts) > 2 else None
                    self.coordinates.append((lat, lon, station_name))
        self._generate_df()
        self.logger.info("API initialized with coordinates.")

    def _generate_df(self) -> None:
        """
        Function to generate a DataFrame from the provided coordinates to match format of GraphConstructer.
        """
        self.logger.info("Generating DataFrame from coordinates...")
        data = []
        for idx, (lat, lon, station_name) in enumerate(self.coordinates):
            # Use provided station_name if available, otherwise generate default name
            name = station_name if station_name else f"Location_{idx + 1}"
            data.append(
                {
                    "odlocation_id": str(idx + 1),
                    "odlocation_name": name,
                    "location_id": str(idx + 1),
                    "network_name": "API_Network",
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        self.data = pd.DataFrame(data)
        self.logger.info("DataFrame generated successfully.")

    def process_coordinates(self, G: nx.Graph):
        """
        Process the provided coordinates and returns metrics per coordinate.
        """
        self.logger.info("Processing coordinates...")

        results = []
        # iterate over each odlocation node in the Graph
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            if node_type == "odlocation":
                name = attrs.get("name", str(node))
                lat = attrs["lat"]
                lon = attrs["lon"]
                landuse = attrs.get("landuse", {})

                poi_weight_sums = attrs.get("poi_weight_sums", {})
                initial_x_decay_x_tfidf_weight = poi_weight_sums.get(
                    "initial_x_decay_x_tfidf_weight", {}
                )

                static_embedding = {}
                # Add per-category static embedding (overall average for each category)
                for category, value in initial_x_decay_x_tfidf_weight.items():
                    static_embedding[category] = {}
                    static_embedding[category]["overall"] = round(float(value), 4)

                poi_time_weight_sums = attrs.get("poi_time_weight_sums", {})
                initial_weight_time_x_decay_x_tfidf_weight = poi_time_weight_sums.get(
                    "initial_weight_time_x_decay_x_tfidf_weight", {}
                )

                # For each category, compute average day and night value (across all days and hours in the respective ranges)
                for (
                    category,
                    days_data,
                ) in initial_weight_time_x_decay_x_tfidf_weight.items():
                    if category not in static_embedding:
                        static_embedding[category] = {"overall": 0.0}
                    day_values = []
                    night_values = []
                    for day, hours_data in days_data.items():
                        for hour_str, weight in hours_data.items():
                            hour = int(hour_str)
                            if 6 <= hour <= 21:
                                day_values.append(float(weight))
                            else:
                                night_values.append(float(weight))
                    # Compute average for each category, or 0 if no values
                    static_embedding[category]["day"] = (
                        round(sum(day_values) / len(day_values), 4)
                        if day_values
                        else 0.0
                    )
                    static_embedding[category]["night"] = (
                        round(sum(night_values) / len(night_values), 4)
                        if night_values
                        else 0.0
                    )

                # Convert time weights to valid JSON format with 2 decimal places
                time_embedding = {}
                for (
                    category,
                    days_data,
                ) in initial_weight_time_x_decay_x_tfidf_weight.items():
                    time_embedding[category] = {}
                    for day, hours_data in days_data.items():
                        time_embedding[category][day] = {
                            int(hour): round(float(weight), 2)
                            for hour, weight in hours_data.items()
                        }

                # get POIs associated with this odlocation via edges
                associated_pois = []
                for neighbor in G.neighbors(node):
                    neighbor_attrs = G.nodes[neighbor]
                    if neighbor_attrs.get("type") == "poi":
                        associated_pois.append(
                            (
                                neighbor_attrs.get("name", str(neighbor)),
                                neighbor_attrs.get("lat", 0),
                                neighbor_attrs.get("lon", 0),
                                neighbor_attrs.get("category", "unknown"),
                                neighbor_attrs.get("entity_name", "unknown"),
                            )
                        )

                # Generate time-based charts for this odlocation
                self.logger.info(f"Generating time-based charts for {name} ({node})...")

                # Create output directory for API charts
                charts_dir = Path("data") / self.config_name / "api" / "charts"
                charts_dir.mkdir(parents=True, exist_ok=True)

                chart_files = {}
                for chart_type in ["inital", "timexdecay", "timexdecayxtfidf"]:
                    # Generate the chart and get the matplotlib figure name should be lat_lon_charttype.png
                    chart_path = (
                        charts_dir / f"{round(lat,6)}_{round(lon,6)}_{chart_type}.png"
                    )

                    # Use visualizer's internal method to get the data and create chart
                    time_based_data = attrs.get("poi_time_weight_sums", {})
                    if chart_type == "inital":
                        time_data = time_based_data.get("initial_weight_time")
                    elif chart_type == "timexdecay":
                        time_data = time_based_data.get(
                            "initial_weight_time_x_decay_weight"
                        )
                    elif chart_type == "timexdecayxtfidf":
                        time_data = time_based_data.get(
                            "initial_weight_time_x_decay_x_tfidf_weight"
                        )

                    if time_data:
                        # Create the chart and save to PNG
                        self._save_time_chart_to_png(
                            node, name, time_data, chart_path, chart_type
                        )
                        chart_files[chart_type] = str(chart_path)
                        self.logger.info(f"Saved {chart_type} chart to {chart_path}")
                    else:
                        chart_files[chart_type] = None

                result = {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "name": name,
                    "type": node_type,
                    "landuse": landuse,
                    "static_embedding": static_embedding,
                    "time_embedding": time_embedding,
                    "associated_pois": associated_pois,
                    "charts": chart_files,
                }

                results.append(result)
                # export to file in data/api/jsons/{lat}_{lon}_results.json
                output_dir = Path("data") / self.config_name / "api" / "jsons"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{round(lat,6)}_{round(lon,6)}_results.json"
                with open(output_file, "w") as f:
                    f.write(json.dumps(result, indent=4))

        # put the result in json
        results = json.dumps(results, indent=4)
        self.logger.info("Coordinates processed successfully.")
        return results

    def _save_time_chart_to_png(
        self,
        node,
        odlocation_name,
        time_data,
        output_path,
        chart_type,
        width=10,
        height=6,
    ):
        """
        Creates and saves a time-based POI weights chart to PNG file.

        Args:
            node: Node ID
            odlocation_name: Name of the odlocation
            time_data: Time-based weight data
            output_path: Path where PNG file will be saved
            chart_type: Type of chart (for title)
            width, height: Size of the chart
        """

        # Create line plot
        fig, ax = plt.subplots(figsize=(width, height))

        # Get all days and sort them by monday -> sunday
        day_order = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        all_days = set()
        for category_data in time_data.values():
            all_days.update(category_data.keys())
        all_days = sorted(all_days, key=lambda day: day_order.index(day))

        # Create x-axis positions (24 hours per day)
        total_hours = len(all_days) * 24
        x_positions = list(range(total_hours))

        # Plot one line per category
        for category, day_data in time_data.items():
            # Concatenate all hour values across days
            category_values = []
            for day in all_days:
                if day in day_data:
                    category_values.extend(day_data[day].values())
                else:
                    category_values.extend([0] * 24)  # Fill with zeros if day missing

            ax.plot(
                x_positions,
                category_values,
                label=f"{category}",
                marker="o",
                markersize=2,
            )

        # Set x-axis labels to show day boundaries
        day_boundaries = [i * 24 for i in range(len(all_days))]
        ax.set_xticks(day_boundaries)
        ax.set_xticklabels(all_days, rotation=45, ha="right")

        # Add vertical lines at day boundaries
        for boundary in day_boundaries[1:-1]:
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.3)

        plt.xlabel("Day and Hour")
        plt.ylabel("POI Weight Sum")
        plt.title(
            f"Time-Based POI Weights for {odlocation_name} ({node}) - {chart_type}"
        )
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()

        # Save to file
        plt.savefig(output_path, format="png", dpi=100)
        plt.close(fig)
