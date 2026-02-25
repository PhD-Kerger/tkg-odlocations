import math

import time
import requests
from .logger import Logger
from pathlib import Path
import numpy as np
import tqdm
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point


class DataPreparation:
    def __init__(
        self,
        max_air_distance_km=0.6,
        max_walking_distance=500,
        osrm_endpoint="http://localhost:5000/route/v1/foot/",
    ):
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )
        self.max_air_distance_km = max_air_distance_km
        self.max_walking_distance = max_walking_distance
        self.osrm_endpoint = osrm_endpoint

    def filter_locations_by_distance(self, odlocations_df, locations_df):
        """
        Filters locations POIs that are within the maximum air distance to a O/D-Location
        Uses BallTree for efficient searching and stores ALL O/D-Locations within the radius
        Sorts O/D-Locations first by air distance

        Args:
            odlocations_df: DataFrame with O/D-Locations
            locations_df: DataFrame with POIs

        Returns:
            DataFrame with filtered Locations and additional columns for all O/D-Locations within radius and their distances
            Dictionary with statistics about the filtering
        """
        self.logger.info(
            f"Filtering locations within {self.max_air_distance_km*1000} meters air distance..."
        )

        # Extract O/D-Location coordinates
        odlocation_coords = np.radians(odlocations_df[["latitude", "longitude"]].values)

        # Extract coordinates of locations to check
        location_coords = np.radians(locations_df[["latitude", "longitude"]].values)

        # Create BallTree based on O/D-Location coordinates
        tree = BallTree(odlocation_coords, metric="haversine")

        # Search for the nearest O/D-Locations for each POI
        # This is only used to calculate the shortest distance to a O/D-Location
        distances, indices = tree.query(location_coords, k=1)

        # Convert distance from radians to kilometers (Earth radius: 6371 km)
        distances_km = distances * 6371.0

        # Create a copy of the DataFrame
        result_df = locations_df.copy()

        # Temporary columns for filtering (will be removed later)
        result_df["nearest_odlocation_idx"] = indices.flatten()
        result_df["air_distance_m_temp"] = np.round(
            distances_km.flatten() * 1000, 2
        )  # Direct conversion to meters with 2 decimal places

        # Initialize columns for lists of all O/D-Locations within radius
        result_df["nearby_odlocation_ids"] = None
        result_df["nearby_odlocation_names"] = None
        result_df["nearby_odlocation_distances"] = None

        # Search for all O/D-Locations within air distance for each location
        max_dist_rad = self.max_air_distance_km / 6371.0  # Convert km to radians

        for idx, loc in result_df.iterrows():
            # Convert coordinates to radians
            loc_coords_single = np.radians(
                np.array([[loc["latitude"], loc["longitude"]]])
            )

            # Find all O/D-Locations within the radius
            nearby_indices = tree.query_radius(loc_coords_single, r=max_dist_rad)[0]

            if len(nearby_indices) > 0:
                # Calculate distances to all O/D-Locations within the radius
                odlocation_data = []

                for odlocation_idx in nearby_indices:
                    odlocation = odlocations_df.iloc[odlocation_idx]

                    # Calculate Haversine distance manually
                    R = 6371000  # Earth radius in meters
                    lat1, lon1 = math.radians(loc["latitude"]), math.radians(
                        loc["longitude"]
                    )
                    lat2, lon2 = math.radians(odlocation["latitude"]), math.radians(
                        odlocation["longitude"]
                    )
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = (
                        math.sin(dlat / 2) ** 2
                        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                    )
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    distance_m = round(R * c, 2)  # Air distance in meters

                    # Store all relevant information in a temporary list
                    odlocation_data.append(
                        {
                            "odlocation_id": int(odlocation["odlocation_id"]),
                            "odlocation_name": odlocation["odlocation_name"],
                            "air_distance": distance_m,
                        }
                    )

                # Sort O/D-Locations by air distance
                sorted_odlocations = sorted(
                    odlocation_data, key=lambda x: x["air_distance"]
                )

                # Extract sorted lists
                nearby_ids = [
                    odlocation["odlocation_id"] for odlocation in sorted_odlocations
                ]
                nearby_names = [
                    odlocation["odlocation_name"] for odlocation in sorted_odlocations
                ]
                nearby_distances_m = [
                    odlocation["air_distance"] for odlocation in sorted_odlocations
                ]

                # Store sorted lists in the DataFrame
                result_df.at[idx, "nearby_odlocation_ids"] = nearby_ids
                result_df.at[idx, "nearby_odlocation_names"] = nearby_names
                result_df.at[idx, "nearby_odlocation_distances"] = nearby_distances_m

        # Filter by maximum air distance to nearest O/D-Location
        filtered_df = result_df[
            result_df["air_distance_m_temp"] <= self.max_air_distance_km * 1000
        ].copy()

        # Remove temporary columns that were only used for filtering
        if "air_distance_m_temp" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["air_distance_m_temp"])

        self.logger.info(
            f"Filtering complete. {len(filtered_df)} locations remain within {self.max_air_distance_km*1000} meters air distance."
        )

        return filtered_df

    def calculate_walking_distances_osrm(
        self, odlocations_df, locations_filtered_by_air
    ):
        """
        Calculates walking distances using local OSRM installation
        Only stores relevant data: lists of nearby O/D-Locations and their distances
        Sorts O/D-Locations by walking distance

        Args:
            odlocations_df: DataFrame with O/D-Locations
            locations_filtered_by_air: DataFrame with locations filtered by air distance

        Returns:
            DataFrame with locations filtered by walking distance
        """
        self.logger.info(
            f"Calculating walking distances with local OSRM installation (maximum {self.max_walking_distance}m)..."
        )

        # Initialize result DataFrame
        result_df = locations_filtered_by_air.copy()

        # Temporary columns for filtering (will be removed later)
        result_df["walking_distance_m_temp"] = float("inf")

        # Initialize lists for all O/D-Locations in radius
        result_df["nearby_walking_distances"] = None
        result_df["nearby_osrm_route_times"] = None

        # Initialize new lists specifically for walking O/D-Locations
        result_df["nearby_walking_odlocation_ids"] = None
        result_df["nearby_walking_odlocation_names"] = None

        # No limitation necessary for local OSRM installation
        sample_locations = locations_filtered_by_air

        # Calculate walking distances with OSRM
        success_count = 0
        error_count = 0

        for idx, loc in tqdm.tqdm(
            sample_locations.iterrows(),
            total=len(sample_locations),
            desc="OSRM Walking Distance Calculation",
        ):
            try:
                # Get the nearest O/D-Location for temporary filtering
                if "nearest_odlocation_idx" in loc:
                    odlocation_idx = int(loc["nearest_odlocation_idx"])
                    odlocation = odlocations_df.iloc[odlocation_idx]

                    # Create URL string for API request to nearest O/D-Location
                    coords = f"{odlocation['longitude']},{odlocation['latitude']};{loc['longitude']},{loc['latitude']}"
                    url = f"{self.osrm_endpoint}{coords}?overview=false"

                    # Send request to local OSRM server for nearest O/D-Location
                    response = requests.get(url)
                    time.sleep(0.01)  # Small delay to avoid overwhelming the server

                    if response.status_code == 200:
                        data = response.json()

                        # Check if a route was found
                        if data["code"] == "Ok" and len(data["routes"]) > 0:
                            # Extract distance in meters and time in seconds
                            distance = data["routes"][0]["distance"]  # Meters

                            # Store result only for filtering
                            result_df.at[idx, "walking_distance_m_temp"] = round(
                                distance, 2
                            )
                            success_count += 1
                        else:
                            # If no route was found, use an approximation (with 2 decimal places)
                            if "air_distance_m_temp" in loc:
                                result_df.at[idx, "walking_distance_m_temp"] = round(
                                    loc["air_distance_m_temp"] * 1.3, 2
                                )  # 30% surcharge
                            elif "air_distance_m" in loc:
                                result_df.at[idx, "walking_distance_m_temp"] = round(
                                    loc["air_distance_m"] * 1.3, 2
                                )  # 30% surcharge
                    else:
                        # For HTTP errors, use an approximation (with 2 decimal places)
                        if "air_distance_m_temp" in loc:
                            result_df.at[idx, "walking_distance_m_temp"] = round(
                                loc["air_distance_m_temp"] * 1.4, 2
                            )  # 40% surcharge
                        elif "air_distance_m" in loc:
                            result_df.at[idx, "walking_distance_m_temp"] = round(
                                loc["air_distance_m"] * 1.4, 2
                            )  # 40% surcharge
                        error_count += 1

                # Now calculate for all O/D-Locations in radius, if available
                if (
                    isinstance(loc["nearby_odlocation_ids"], list)
                    and len(loc["nearby_odlocation_ids"]) > 0
                ):
                    # Collect all O/D-Location information in a temporary list with air distance
                    odlocation_data = []

                    # For each nearby O/D-Location, calculate the walking distance
                    for i, odlocation_id in enumerate(loc["nearby_odlocation_ids"]):
                        # Find O/D-Location in DataFrame
                        odlocation_row = odlocations_df[
                            odlocations_df["odlocation_id"] == str(odlocation_id)
                        ]

                        if len(odlocation_row) > 0:
                            odlocation_data_item = {
                                "odlocation_id": str(odlocation_id),
                                "odlocation_name": loc["nearby_odlocation_names"][i],
                                "air_distance": loc["nearby_odlocation_distances"][i],
                                "walking_distance": None,
                                "osrm_route_time": None,
                            }

                            odlocation_data.append(odlocation_data_item)
                            odlocation_data_item = odlocation_data[
                                -1
                            ]  # Reference to last element

                            odlocation_data_real = odlocation_row.iloc[0]

                            # Create URL string for API request
                            coords = f"{odlocation_data_real['longitude']},{odlocation_data_real['latitude']};{loc['longitude']},{loc['latitude']}"
                            url = f"{self.osrm_endpoint}{coords}?overview=false"

                            try:
                                # Send request to local OSRM server
                                response = requests.get(url)
                                time.sleep(
                                    0.01
                                )  # Small delay to avoid overwhelming the server

                                if response.status_code == 200:
                                    data = response.json()

                                    # Check if a route was found
                                    if data["code"] == "Ok" and len(data["routes"]) > 0:
                                        # Extract distance in meters and time in seconds
                                        distance = data["routes"][0][
                                            "distance"
                                        ]  # Meters
                                        duration = data["routes"][0][
                                            "duration"
                                        ]  # Seconds

                                        # Store values in odlocation data list
                                        odlocation_data_item["walking_distance"] = (
                                            round(distance, 2)
                                        )
                                        odlocation_data_item["osrm_route_time"] = round(
                                            duration, 2
                                        )
                                    else:
                                        # If no route was found, use approximation from air distance
                                        air_distance = loc[
                                            "nearby_odlocation_distances"
                                        ][i]
                                        odlocation_data_item["walking_distance"] = (
                                            round(air_distance * 1.3, 2)
                                        )  # 30% surcharge
                                        odlocation_data_item["osrm_route_time"] = round(
                                            (air_distance * 1.3) / 1.4, 2
                                        )
                                else:
                                    # For HTTP errors, use approximation from air distance
                                    air_distance = loc["nearby_odlocation_distances"][i]
                                    odlocation_data_item["walking_distance"] = round(
                                        air_distance * 1.4, 2
                                    )  # 40% surcharge
                                    odlocation_data_item["osrm_route_time"] = round(
                                        (air_distance * 1.4) / 1.4, 2
                                    )

                            except Exception as e:
                                # For errors, use conservative estimate from air distance
                                air_distance = loc["nearby_odlocation_distances"][i]
                                odlocation_data_item["walking_distance"] = round(
                                    air_distance * 1.5, 2
                                )  # 50% surcharge
                                odlocation_data_item["osrm_route_time"] = round(
                                    (air_distance * 1.5) / 1.4, 2
                                )

                    # Sort odlocations first by walking distance, then by air distance
                    sorted_odlocations = sorted(
                        odlocation_data,
                        key=lambda x: (x["walking_distance"], x["air_distance"]),
                    )

                    # Extract sorted lists for all odlocations (for nearby_odlocation_ids, names, distances)
                    nearby_ids = [
                        odlocation["odlocation_id"] for odlocation in sorted_odlocations
                    ]
                    nearby_names = [
                        odlocation["odlocation_name"]
                        for odlocation in sorted_odlocations
                    ]
                    nearby_air_distances = [
                        odlocation["air_distance"] for odlocation in sorted_odlocations
                    ]

                    # Store lists for all odlocations in DataFrame
                    result_df.at[idx, "nearby_odlocation_ids"] = nearby_ids
                    result_df.at[idx, "nearby_odlocation_names"] = nearby_names
                    result_df.at[idx, "nearby_odlocation_distances"] = (
                        nearby_air_distances
                    )

                    # Filter odlocations within walking distance for separate walking lists
                    walking_odlocations = [
                        odlocation
                        for odlocation in sorted_odlocations
                        if odlocation["walking_distance"] <= self.max_walking_distance
                    ]

                    if walking_odlocations:
                        # Create lists only for odlocations within maximum walking distance
                        nearby_walking_odlocation_ids = [
                            odlocation["odlocation_id"]
                            for odlocation in walking_odlocations
                        ]
                        nearby_walking_odlocation_names = [
                            odlocation["odlocation_name"]
                            for odlocation in walking_odlocations
                        ]
                        nearby_walking_distances = [
                            odlocation["walking_distance"]
                            for odlocation in walking_odlocations
                        ]
                        nearby_osrm_route_times = [
                            odlocation["osrm_route_time"]
                            for odlocation in walking_odlocations
                        ]

                        # Store lists in DataFrame
                        result_df.at[idx, "nearby_walking_odlocation_ids"] = (
                            nearby_walking_odlocation_ids
                        )
                        result_df.at[idx, "nearby_walking_odlocation_names"] = (
                            nearby_walking_odlocation_names
                        )
                        result_df.at[idx, "nearby_walking_distances"] = (
                            nearby_walking_distances
                        )
                        result_df.at[idx, "nearby_osrm_route_times"] = (
                            nearby_osrm_route_times
                        )
                    else:
                        # If no odlocations are within walking distance, set empty lists
                        result_df.at[idx, "nearby_walking_odlocation_ids"] = []
                        result_df.at[idx, "nearby_walking_odlocation_names"] = []
                        result_df.at[idx, "nearby_walking_distances"] = []
                        result_df.at[idx, "nearby_osrm_route_times"] = []

            except Exception as e:
                self.logger.error(
                    f"Error at coordinates {loc['latitude']}, {loc['longitude']}: {str(e)}"
                )
                # For errors, use conservative estimate (with 2 decimal places)
                if "air_distance_m_temp" in loc:
                    result_df.at[idx, "walking_distance_m_temp"] = round(
                        loc["air_distance_m_temp"] * 1.5, 2
                    )  # 50% surcharge
                elif "air_distance_m" in loc:
                    result_df.at[idx, "walking_distance_m_temp"] = round(
                        loc["air_distance_m"] * 1.5, 2
                    )  # 50% surcharge
                error_count += 1

        self.logger.info(
            f"Successful OSRM requests: {success_count}, Errors: {error_count}"
        )

        # For uncalculated entries (if we sampled), use approximation (with 2 decimal places)
        if len(sample_locations) < len(locations_filtered_by_air):
            missing_indices = set(locations_filtered_by_air.index) - set(
                sample_locations.index
            )
            for idx in missing_indices:
                if (
                    "air_distance_m_temp" in result_df.columns
                    and idx in result_df.index
                ):
                    result_df.at[idx, "walking_distance_m_temp"] = round(
                        result_df.loc[idx, "air_distance_m_temp"] * 1.3, 2
                    )
                elif "air_distance_m" in result_df.columns and idx in result_df.index:
                    result_df.at[idx, "walking_distance_m_temp"] = round(
                        result_df.loc[idx, "air_distance_m"] * 1.3, 2
                    )

        # Filter by maximum walking distance
        filtered_df = result_df[
            result_df["walking_distance_m_temp"] <= self.max_walking_distance
        ].copy()

        # Remove temporary column for filtering
        if "walking_distance_m_temp" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["walking_distance_m_temp"])

        # Calculate number of nearby odlocations with walking distance below maximum
        filtered_df["nearby_walking_odlocation_count"] = filtered_df[
            "nearby_walking_odlocation_ids"
        ].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # Remove other unnecessary columns
        columns_to_drop = []
        for col in filtered_df.columns:
            if col in [
                "nearest_odlocation_idx",
                "nearest_odlocation_id",
                "nearest_odlocation_name",
                "air_distance_m",
                "air_distance_m_temp",
                "walking_distance_m",
                "osrm_route_time_s",
            ]:
                columns_to_drop.append(col)

        if columns_to_drop:
            filtered_df = filtered_df.drop(columns=columns_to_drop)

        return filtered_df

    def parse_opening_hours_pois(self, pois_df):
        """
        Parses the opening hours of POIs and adds structured information to the DataFrame.

        Args:
            pois_df (pd.DataFrame): DataFrame containing points of interest data.

        Returns:
            dict: Dictionary with POI IDs as keys and opening hours data as values.
                Each entry contains hourly open/closed status for each weekday and PH status.
        """
        self.logger.info("Parsing opening hours for POIs...")

        if pois_df is None or pois_df.empty:
            self.logger.warning("POIs DataFrame is empty or None.")
            return None

        # get opening hours column
        if "opening_hours" not in pois_df.columns:
            self.logger.warning("No 'opening_hours' column found in POIs DataFrame.")
            return None

        opening_hours_data = {}

        # Day name mapping
        day_mapping = {
            "Mo": "monday",
            "Tu": "tuesday",
            "We": "wednesday",
            "Th": "thursday",
            "Fr": "friday",
            "Sa": "saturday",
            "Su": "sunday",
            "Di": "tuesday",  # German abbreviation for Tuesday
            "Mi": "wednesday",  # German abbreviation for Wednesday
            "Do": "thursday",  # German abbreviation for Thursday
            "So": "sunday",  # German abbreviation for Sunday
        }

        for idx, poi in pois_df.iterrows():
            hours = poi.get("opening_hours", "")
            poi_id = poi.get("poi_id", idx)
            poi_category = poi.get("category", "unknown")

            if not hours or pd.isna(hours):
                self.logger.warning(
                    f"Missing opening hours for POI ID {poi_id} in category {poi_category}. Using default schedule."
                )
            # Initialize structure based on category:
            # food_beverage: assume typical hours if missing: mon-fri 08:00-20:00, sat-sun 10:00-18:00
            if poi_category == "food_beverage":
                schedule = {
                    "monday": [0] * 8 + [1] * 12 + [0] * 4,
                    "tuesday": [0] * 8 + [1] * 12 + [0] * 4,
                    "wednesday": [0] * 8 + [1] * 12 + [0] * 4,
                    "thursday": [0] * 8 + [1] * 12 + [0] * 4,
                    "friday": [0] * 8 + [1] * 12 + [0] * 4,
                    "saturday": [0] * 10 + [1] * 8 + [0] * 6,
                    "sunday": [0] * 10 + [1] * 8 + [0] * 6,
                }
            # retail_shopping: assume typical hours if missing: mon-sat 09:00-20:00, sun closed
            elif poi_category in ["retail_shopping", "services"]:
                schedule = {
                    "monday": [0] * 9 + [1] * 11 + [0] * 4,
                    "tuesday": [0] * 9 + [1] * 11 + [0] * 4,
                    "wednesday": [0] * 9 + [1] * 11 + [0] * 4,
                    "thursday": [0] * 9 + [1] * 11 + [0] * 4,
                    "friday": [0] * 9 + [1] * 11 + [0] * 4,
                    "saturday": [0] * 9 + [1] * 11 + [0] * 4,
                    "sunday": [0] * 24,
                }
            # transportation_car and transportation_railway 24/7 open
            elif poi_category in ["transportation_car", "transportation_railway"]:
                schedule = {
                    "monday": [1] * 24,
                    "tuesday": [1] * 24,
                    "wednesday": [1] * 24,
                    "thursday": [1] * 24,
                    "friday": [1] * 24,
                    "saturday": [1] * 24,
                    "sunday": [1] * 24,
                }
                opening_hours_data[poi_id] = schedule
                continue
            # education and entertainment: assume typical hours if missing: mon-sun 08:00-20:00
            elif poi_category in ["education", "entertainment"]:
                schedule = {
                    "monday": [0] * 8 + [1] * 12 + [0] * 4,
                    "tuesday": [0] * 8 + [1] * 12 + [0] * 4,
                    "wednesday": [0] * 8 + [1] * 12 + [0] * 4,
                    "thursday": [0] * 8 + [1] * 12 + [0] * 4,
                    "friday": [0] * 8 + [1] * 12 + [0] * 4,
                    "saturday": [0] * 8 + [1] * 12 + [0] * 4,
                    "sunday": [0] * 8 + [1] * 12 + [0] * 4,
                }
            else:
                # Default: all open
                schedule = {
                    "monday": [1] * 24,
                    "tuesday": [1] * 24,
                    "wednesday": [1] * 24,
                    "thursday": [1] * 24,
                    "friday": [1] * 24,
                    "saturday": [1] * 24,
                    "sunday": [1] * 24,
                }

            # Split by semicolon for different rules
            rules = hours.split(";")

            for rule in rules:
                rule = rule.strip()

                # Parse day ranges and times
                # Handle complex cases like "Mo-Fr 08:00-12:30, Mo,Fr 14:00-16:00"
                # Split by comma to get individual day-time segments
                segments = rule.split(",")

                for segment in segments:
                    segment = segment.strip()

                    try:
                        # Find the separator between days and times
                        day_part = None
                        time_part = None

                        # Split by spaces and find where the time pattern starts
                        parts = segment.split()
                        for i, part in enumerate(parts):
                            # Check if this part looks like a time (contains :)
                            if ":" in part:
                                # Everything before this is the day part
                                day_part = " ".join(parts[:i])
                                time_part = " ".join(parts[i:])
                                break

                        if not day_part or not time_part:
                            continue

                        # Parse days
                        days = []
                        if "-" in day_part and "," in day_part:
                            # Mixed case: comma-separated items where some might be ranges (e.g., "Mo,We-Su")
                            day_items = [item.strip() for item in day_part.split(",")]
                            for item in day_items:
                                if "-" in item:
                                    # This item is a range
                                    start_day, end_day = [
                                        d.strip() for d in item.split("-")
                                    ]
                                    day_keys = list(day_mapping.keys())
                                    if start_day in day_keys and end_day in day_keys:
                                        start_idx = day_keys.index(start_day)
                                        end_idx = day_keys.index(end_day)
                                        for i in range(start_idx, end_idx + 1):
                                            day_name = day_mapping[day_keys[i]]
                                            if day_name not in days:
                                                days.append(day_name)
                                else:
                                    # Single day
                                    if item in day_mapping:
                                        day_name = day_mapping[item]
                                        if day_name not in days:
                                            days.append(day_name)
                        elif "-" in day_part:
                            # Day range (e.g., "Mo-Th")
                            start_day, end_day = [
                                d.strip() for d in day_part.split("-")
                            ]
                            day_keys = list(day_mapping.keys())
                            if start_day in day_keys and end_day in day_keys:
                                start_idx = day_keys.index(start_day)
                                end_idx = day_keys.index(end_day)
                                for i in range(start_idx, end_idx + 1):
                                    days.append(day_mapping[day_keys[i]])
                        elif "," in day_part:
                            # Multiple days (e.g., "Fr,Sa" or "Fr, Sa")
                            for day in day_part.split(","):
                                day = day.strip()
                                if day in day_mapping:
                                    days.append(day_mapping[day])
                        else:
                            # Single day
                            day_part = day_part.strip()
                            if day_part in day_mapping:
                                days.append(day_mapping[day_part])

                        # Parse time range - now should only be a single time range per segment
                        if "-" in time_part:
                            time_parts = time_part.split("-")
                            if len(time_parts) == 2:
                                start_time, end_time = time_parts
                                start_hour = int(start_time.split(":")[0])
                                end_hour = int(end_time.split(":")[0])

                                # Handle opening hours that span past midnight
                                if end_hour <= start_hour:
                                    # Opens before midnight, closes after midnight
                                    day_order = [
                                        "monday",
                                        "tuesday",
                                        "wednesday",
                                        "thursday",
                                        "friday",
                                        "saturday",
                                        "sunday",
                                    ]

                                    for day in days:
                                        # Mark hours from start_hour to 24:00 as open on current day
                                        for hour in range(start_hour, 24):
                                            schedule[day][hour] = 1

                                        # Mark hours from 00:00 to end_hour as open on next day
                                        next_day_idx = (day_order.index(day) + 1) % 7
                                        next_day = day_order[next_day_idx]
                                        for hour in range(0, end_hour):
                                            schedule[next_day][hour] = 1
                                else:
                                    # Normal case: opens and closes on the same day
                                    for day in days:
                                        for hour in range(start_hour, end_hour):
                                            if hour < 24:
                                                schedule[day][hour] = 1

                    except Exception as e:
                        self.logger.warning(
                            f"Could not parse segment '{segment}' for POI {poi_id}: {str(e)}"
                        )
                        continue

            opening_hours_data[poi_id] = schedule
        # export to csv
        opening_hours_df = pd.DataFrame.from_dict(opening_hours_data, orient="index")
        opening_hours_df.index.name = "poi_id"
        opening_hours_df.reset_index(inplace=True)
        pois_df = pois_df.merge(opening_hours_df, on="poi_id", how="left")
        # drop poi_id,location_id,poi_name,entity_name,cuisine,longitude,latitude,city,postal_code,state,country,category,
        pois_df = pois_df.drop(
            columns=[
                "location_id",
                "poi_name",
                "entity_name",
                "cuisine",
                "longitude",
                "latitude",
                "city",
                "postal_code",
                "state",
                "country",
                "category",
            ]
        )
        self.logger.info(f"Parsed opening hours for {len(opening_hours_data)} POIs")
        return opening_hours_data

    def odlocation_landuse(self, odlocation_df, landuse_gdf, radius_m=500):
        """
        Calculate landuse classification features for O/D-Locations.
        Args:
            odlocation_df (pd.DataFrame): DataFrame containing O/D-Locations with 'latitude' and 'longitude' columns.
            landuse_gdf (gpd.GeoDataFrame): GeoDataFrame containing land use data.
            radius_m (int): Radius in meters
        Returns:
            pd.DataFrame: DataFrame with added landuse features.
        """
        self.logger.info("Calculating landuse features for O/D-Locations...")

        # Collect all features in a list for better performance
        all_features = []

        for idx, odlocation in tqdm.tqdm(
            odlocation_df.iterrows(),
            total=len(odlocation_df),
            desc="Calculating landuse features for O/D-Locations",
        ):
            landuse_features = self.get_landuse_features(
                odlocation["latitude"],
                odlocation["longitude"],
                landuse_gdf,
                radius_m=radius_m,
            )
            all_features.append(landuse_features)

        # Create DataFrame from features and concatenate with original
        features_df = pd.DataFrame(all_features, index=odlocation_df.index)
        result_df = pd.concat([odlocation_df, features_df], axis=1)

        return result_df

    def get_landuse_features(self, latitude, longitude, landuse_gdf, radius_m):
        """
        Get landuse classification features for a single point.

        Args:
            latitude (float): Latitude of the point.
            longitude (float): Longitude of the point.
            landuse_gdf (gpd.GeoDataFrame): GeoDataFrame containing land use data.
            radius_m (int): Radius in meters for the circular area around the point.

        Returns:
            dict: Dictionary with landuse percentages and classification for the point.
        """

        # Create point geometry
        point = Point(longitude, latitude)
        point_gdf = gpd.GeoDataFrame([{"geometry": point}], crs="EPSG:4326")

        # Set CRS if not already set for landuse
        if landuse_gdf.crs is None:
            landuse_gdf = landuse_gdf.set_crs("EPSG:4326")

        # Ensure same CRS
        if point_gdf.crs != landuse_gdf.crs:
            landuse_gdf = landuse_gdf.to_crs(point_gdf.crs)

        # Project to a metric CRS for accurate area calculation (EPSG:32632 for UTM Zone 32N)
        point_projected = point_gdf.to_crs("EPSG:32632")
        landuse_projected = landuse_gdf.to_crs("EPSG:32632")

        # Create circular buffer around point
        circle = point_projected.iloc[0].geometry.buffer(radius_m)
        total_area = circle.area

        # Find intersecting landuse polygons
        intersecting = landuse_projected[landuse_projected.intersects(circle)]

        # Calculate landuse areas
        landuse_areas = {}
        for _, landuse in intersecting.iterrows():
            intersection = landuse.geometry.intersection(circle)
            if not intersection.is_empty:
                landuse_type = landuse["landuse"]
                intersection_area = intersection.area

                if landuse_type in landuse_areas:
                    landuse_areas[landuse_type] += intersection_area
                else:
                    landuse_areas[landuse_type] = intersection_area

        # Define landuse categories
        categories = {
            "work": ["commercial", "industrial", "construction", "retail"],
            "residential": ["residential"],
            "leisure": [
                "recreation_ground",
                "forest",
                "meadow",
                "grass",
                "greenfield",
                "allotments",
                "cemetery",
                "village_green",
                "orchard",
                "flowerbed",
            ],
            "mixed_use": ["farmland", "farmyard", "religious", "railway"],
            "other": ["landfill", "reservoir", "greenhouse_horticulture"],
        }

        # Categorize and aggregate
        category_percentages = {
            "work": 0.0,
            "residential": 0.0,
            "leisure": 0.0,
            "mixed_use": 0.0,
            "other": 0.0,
        }

        for landuse_type, area in landuse_areas.items():
            percentage = (area / total_area) * 100

            # Find category for this landuse type
            category_found = False
            for category, types in categories.items():
                if landuse_type in types:
                    category_percentages[category] += percentage
                    category_found = True
                    break

            if not category_found:
                category_percentages["other"] += percentage

        # Determine classification
        work_score = category_percentages["work"]
        residential_score = category_percentages["residential"]
        leisure_score = category_percentages["leisure"]

        if work_score > 30:
            if residential_score > 20:
                classification = "mixed_work_residential"
            else:
                classification = "work_dominant"
        elif residential_score > 30:
            if work_score > 10:
                classification = "residential_with_work"
            else:
                classification = "residential_dominant"
        elif leisure_score > 30:
            classification = "leisure_dominant"
        else:
            classification = "mixed_use"

        # Return results as percentages (0-1 range)
        return {
            "work_percentage": round(work_score / 100, 4),
            "residential_percentage": round(residential_score / 100, 4),
            "leisure_percentage": round(leisure_score / 100, 4),
            "mixed_use_percentage": round(category_percentages["mixed_use"] / 100, 4),
            "other_percentage": round(category_percentages["other"] / 100, 4),
            "classification": classification,
        }
