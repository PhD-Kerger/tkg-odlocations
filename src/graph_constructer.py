import networkx as nx
from src import Logger
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict


class GraphConstructer:
    def __init__(
        self,
        odlocations_df,
        pois_df,
        pois_opening_hours,
        private_cap_threshold=0.75,
    ):
        self.G = nx.Graph()
        self.odlocations_df = odlocations_df
        self.pois_df = pois_df
        self.pois_opening_hours = pois_opening_hours
        self.private_cap_threshold = private_cap_threshold

        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def construct_graph(self):
        """
        Constructs the graph G by adding odlocations and POIs as nodes.
        """
        self.logger.info("Starting graph construction")

        self._add_odlocations_to_graph()
        self._add_pois_to_graph()

        if "classification" in self.odlocations_df.columns:
            self._add_landuse_classification_to_graph()
            self._add_private_scores_to_graph()

        tfidf_pois_normalized = self._calculate_tf_idf_for_pois()

        self._add_poi_edges(tfidf_pois_normalized)

        self._update_nodes_after_weights()

        self.logger.info("Graph construction completed.")

    def _add_odlocations_to_graph(self):
        """
        Adds odlocation nodes to the graph G.
        """
        for _, odlocation in self.odlocations_df.iterrows():
            node_id = odlocation["odlocation_id"]
            if not self.G.has_node(node_id):
                self.G.add_node(
                    node_id,
                    type="odlocation",
                    name=odlocation["odlocation_name"],
                    network=odlocation["network_name"],
                    lat=odlocation["latitude"],
                    lon=odlocation["longitude"],
                    location_id=odlocation["location_id"],
                )
            else:
                self.logger.warning(
                    f"Fount duplicate odlocation_id: {node_id}, Odlocation name: {odlocation['odlocation_name']}"
                )
        self.logger.info(
            f"Successfully added {self.odlocations_df.shape[0]} odlocation nodes to the graph."
        )

    def _add_pois_to_graph(self):
        """
        Adds POI nodes to the graph G.
        POI nodes are added with unique identifiers based on poi_id.
        """
        # Add POI nodes - create a separate node for each POI, even if location_id is the same
        successful_poi_nodes = 0
        duplicate_poi_nodes = 0
        for idx, poi in self.pois_df.iterrows():
            # Use poi_id as a unique identifier for each POI
            node_id = f"P_{poi['poi_id']}"

            # Check for duplicate poi_id
            if not self.G.has_node(node_id):
                # Create the node with all important attributes
                self.G.add_node(
                    node_id,
                    type="poi",
                    poi_id=poi["poi_id"],
                    location_id=poi["location_id"],
                    name=poi["poi_name"],
                    entity_name=poi["entity_name"],
                    category=poi["category"],
                    lat=poi["latitude"],
                    lon=poi["longitude"],
                )

                # Add parsed opening hours dict
                if poi["poi_id"] in self.pois_opening_hours:
                    self.G.nodes[node_id]["opening_hours"] = self.pois_opening_hours[
                        poi["poi_id"]
                    ]

                # Add lists of nearby odlocations in air distance mode
                if "nearby_odlocation_ids" in poi and isinstance(
                    poi["nearby_odlocation_ids"], list
                ):
                    self.G.nodes[node_id]["nearby_odlocation_ids"] = poi[
                        "nearby_odlocation_ids"
                    ]
                    self.G.nodes[node_id]["nearby_odlocation_names"] = poi[
                        "nearby_odlocation_names"
                    ]
                    self.G.nodes[node_id]["nearby_odlocation_distances"] = poi[
                        "nearby_odlocation_distances"
                    ]

                # Add lists of nearby odlocations in walking mode
                if "nearby_walking_odlocation_ids" in poi and isinstance(
                    poi["nearby_walking_odlocation_ids"], list
                ):
                    self.G.nodes[node_id]["nearby_walking_odlocation_ids"] = poi[
                        "nearby_walking_odlocation_ids"
                    ]
                    self.G.nodes[node_id]["nearby_walking_odlocation_names"] = poi[
                        "nearby_walking_odlocation_names"
                    ]
                    self.G.nodes[node_id]["nearby_walking_distances"] = poi[
                        "nearby_walking_distances"
                    ]
                    self.G.nodes[node_id]["nearby_osrm_route_times"] = poi[
                        "nearby_osrm_route_times"
                    ]

                successful_poi_nodes += 1
            else:
                self.logger.warning(
                    f"Duplicate poi_id found: {poi['poi_id']}, POI name: {poi['poi_name']}"
                )
                duplicate_poi_nodes += 1
        self.logger.info(
            f"Successfully added {successful_poi_nodes} POI nodes to the graph."
        )

    def _add_landuse_classification_to_graph(self):
        """
        Adds to each odlocation node land use classification information from the odlocations_df DataFrame.
        The landuse features should already be calculated using the radius-based approach.
        """
        self.logger.info(
            "Adding land use classification information to odlocation nodes."
        )

        landuse_added_count = 0

        for _, odlocation in self.odlocations_df.iterrows():
            odlocation_node_id = odlocation["odlocation_id"]

            # Add land use classification info from DataFrame columns
            self.G.nodes[odlocation_node_id]["landuse"] = {
                "work_percentage": odlocation.get("work_percentage", None),
                "residential_percentage": odlocation.get(
                    "residential_percentage", None
                ),
                "leisure_percentage": odlocation.get("leisure_percentage", None),
                "mixed_use_percentage": odlocation.get("mixed_use_percentage", None),
                "other_percentage": odlocation.get("other_percentage", None),
                "classification": odlocation.get("classification", None),
            }
            landuse_added_count += 1

        self.logger.info(
            f"Land use classification information added to {landuse_added_count} odlocation nodes."
        )

    def _add_private_scores_to_graph(self):
        """
        Calculates and adds private scores to odlocation nodes based on landuse data.
        Private score combines residential and work percentages to represent private trip likelihood.
        """
        self.logger.info("Calculating private scores for odlocation nodes...")

        private_scores_added = 0

        for node in self.G.nodes():
            node_attrs = self.G.nodes[node]
            if node_attrs.get("type") == "odlocation":
                # Check if landuse info exists
                if "landuse" in node_attrs:
                    landuse = node_attrs["landuse"]
                    residential_pct = landuse.get("residential_percentage", 0)
                    work_pct = landuse.get("work_percentage", 0)

                    # Calculate static private score (weighted average)
                    # Higher weight on residential (0.6) since home trips are more universally private
                    static_private_score = 0.6 * residential_pct + 0.4 * work_pct

                    # Store the private score and components
                    node_attrs["private_score"] = {
                        "static": static_private_score,
                        "residential_component": residential_pct,
                        "work_component": work_pct,
                    }

                    # Create time-dependent private scores (24 hours x 7 days)
                    time_private_scores = {}
                    day_names = [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ]

                    for day in day_names:
                        time_private_scores[day] = []
                        is_weekend = day in ["saturday", "sunday"]

                        for hour in range(24):
                            if is_weekend:
                                # Weekends: primarily residential-based, minimal work influence
                                hour_score = 0.9 * residential_pct + 0.1 * work_pct
                            else:
                                # Weekdays: time-dependent weighting
                                if 6 <= hour < 9:  # Morning commute: emphasize work
                                    hour_score = 0.3 * residential_pct + 0.7 * work_pct
                                elif (
                                    17 <= hour < 20
                                ):  # Evening commute: emphasize residential
                                    hour_score = 0.8 * residential_pct + 0.2 * work_pct
                                elif (
                                    9 <= hour < 17
                                ):  # Work hours: balanced but work-leaning
                                    hour_score = 0.4 * residential_pct + 0.6 * work_pct
                                elif (
                                    20 <= hour < 24 or hour < 6
                                ):  # Night/early morning: residential dominant
                                    hour_score = (
                                        0.85 * residential_pct + 0.15 * work_pct
                                    )
                                else:  # Other times: balanced
                                    hour_score = 0.5 * residential_pct + 0.5 * work_pct

                            time_private_scores[day].append(hour_score)

                    node_attrs["private_score"]["time_dependent"] = time_private_scores
                    private_scores_added += 1

        self.logger.info(
            f"Added private scores to {private_scores_added} odlocation nodes."
        )

    def get_graph(self):
        """
        Returns the constructed graph G.
        """
        return self.G

    def _add_poi_edges(self, poi_type_normalized, max_distance=500):
        """
        Adds edges between odlocations and POIs with fixed weights of 1.0.
        Saves the resulting graph in both Pickle and GraphML formats.
        Also provides statistics on the added edges.
        """
        # Counter for added edges
        added_edges_odlocation_poi = 0

        self.logger.info(
            "Adding edges between odlocations and POIs with fixed weights..."
        )
        # Connect odlocations with POIs
        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") == "poi":
                # Priorisiere walking_distances, wenn verfügbar
                if "nearby_walking_odlocation_ids" in attrs and isinstance(
                    attrs["nearby_walking_odlocation_ids"], list
                ):
                    odlocation_ids = attrs["nearby_walking_odlocation_ids"]
                    odlocation_distances = attrs["nearby_walking_distances"]
                    distance_type = "walking"
                elif "nearby_odlocation_ids" in attrs and isinstance(
                    attrs["nearby_odlocation_ids"], list
                ):
                    odlocation_ids = attrs["nearby_odlocation_ids"]
                    odlocation_distances = attrs["nearby_odlocation_distances"]
                    distance_type = "straight"
                else:
                    continue

                # Füge Kanten für jede nahe Odlocation hinzu
                for i, odlocation_id in enumerate(odlocation_ids):
                    if i < len(odlocation_distances):  # Sicherheitscheck
                        distance = odlocation_distances[i]

                        # Nur Kanten hinzufügen, wenn die Entfernung unter dem Schwellenwert liegt
                        if distance <= max_distance:
                            node_category = self._get_node_category_from_graph(node)
                            decay = self._distance_to_weight_gaussian(distance)
                            tfidf = poi_type_normalized.get(node_category, 1.0)

                            # 1st try with simple initial weight
                            initial_weight = 1.0
                            # 2nd try with decay by distance
                            initial_x_decay_weight = initial_weight * decay
                            # 3rd try with decay and tfidf
                            initial_x_decay_x_tfidf_weight = (
                                initial_x_decay_weight * tfidf
                            )
                            # 4th try with opening hours consideration
                            initial_weight_time = attrs.get("opening_hours")

                            # Apply decay directly to opening_hours
                            initial_weight_time_x_decay_weight = {}
                            if initial_weight_time:
                                for day, hours in initial_weight_time.items():
                                    if isinstance(hours, list):
                                        initial_weight_time_x_decay_weight[day] = [
                                            h * decay for h in hours
                                        ]
                                    else:
                                        initial_weight_time_x_decay_weight[day] = (
                                            hours * decay
                                        )

                            # Apply tfidf to the decay-adjusted weights
                            initial_weight_time_x_decay_x_tfidf_weight = {}
                            for (
                                day,
                                hours,
                            ) in initial_weight_time_x_decay_weight.items():
                                if isinstance(hours, list):
                                    initial_weight_time_x_decay_x_tfidf_weight[day] = [
                                        h * tfidf for h in hours
                                    ]
                                else:
                                    initial_weight_time_x_decay_x_tfidf_weight[day] = (
                                        hours * tfidf
                                    )

                            # Füge die Kante mit Attributen hinzu
                            self.G.add_edge(
                                odlocation_id,
                                node,
                                initial_weight=initial_weight,
                                initial_x_decay_weight=initial_x_decay_weight,
                                initial_x_decay_x_tfidf_weight=initial_x_decay_x_tfidf_weight,
                                initial_weight_time=initial_weight_time,
                                initial_weight_time_x_decay_weight=initial_weight_time_x_decay_weight,
                                initial_weight_time_x_decay_x_tfidf_weight=initial_weight_time_x_decay_x_tfidf_weight,
                                decay=decay,
                                distance=distance,
                                distance_type=distance_type,
                                edge_type="odlocation_to_poi",
                            )
                            added_edges_odlocation_poi += 1

        self.logger.info(
            f"Successfully added {added_edges_odlocation_poi} O/D-Location-POI weighted edges."
        )

    def _distance_to_weight_gaussian(self, distance, sigma=100):
        """Gaussian function to convert distance to weight.

        Args:
            distance: Distance in meters
            sigma: Parameter to control the rate of decrease (default: 200)

        Returns:
            Weight between 0 and 1
        """
        return np.exp(-(distance**2) / (2 * sigma**2))

    def _calculate_tf_idf_for_pois(self):
        """
        Calculates IDF statistics for POI types in the graph G.
        Returns a DataFrame with POI types, their counts, and normalized IDF values
        """
        poi_types = Counter()

        for node in self.G.nodes():
            node_attrs = self.G.nodes[node]
            if node_attrs.get("type") == "poi":
                poi_category = node_attrs.get(
                    "category", node_attrs.get("name", "unknown")
                )
                poi_types[poi_category] += 1

        total_pois = sum(poi_types.values())
        self.logger.info(
            f"Calculating IDF statistics for POI types. Found {total_pois} POIs with {len(poi_types)} unique types."
        )

        poi_counts = pd.DataFrame(
            list(poi_types.items()), columns=["category", "count"]
        )

        # Calculate IDF
        num_categories = len(poi_counts)
        idf = np.log(
            num_categories / (poi_counts["count"] + 1)
        )  # +1 to avoid division by zero

        # Normalize IDF to range [0.05, 1]
        if idf.max() != idf.min():  # Check if there is any range
            idf = (idf - idf.min()) / (idf.max() - idf.min())
            idf = idf * 0.95 + 0.05  # Scale to [0.05, 1]
        else:
            idf = (
                np.ones_like(idf) * 0.5
            )  # If all values are the same, set to the mean value
        # Create a dictionary for the IDF values
        poi_type_normalized = defaultdict(float)
        for i, row in poi_counts.iterrows():
            poi_type = row["category"]
            poi_type_normalized[poi_type] = float(idf[i])

        return poi_type_normalized

    # Füge eine Hilfsfunktion hinzu, um IDF-Werte für einen Knoten zu erhalten
    def _get_node_category_from_graph(self, node_id):
        """Get the category of a node from the graph

        Args:
            node_id: ID of the node
        """
        if node_id in self.G.nodes:
            node = self.G.nodes[node_id]
            return node.get("category", node.get("name", str(node_id)))
        return None

    def _update_nodes_after_weights(self):
        """
        Updates node attributes based on the weights of connected edges.
        Specifically, it calculates sums of weights for connected POIs per category.
        """
        # First pass: collect distribution of total POI weights per odlocation for percentile-based penalties
        weight_distributions = {}
        for weight_key in [
            "initial_weight",
            "initial_x_decay_weight",
            "initial_x_decay_x_tfidf_weight",
        ]:
            total_weights = []
            for node in self.G.nodes():
                node_attrs = self.G.nodes[node]
                if node_attrs.get("type") == "odlocation":
                    connected_edges = self.G.edges(node, data=True)
                    temp_sum = 0.0
                    for u, v, edge_attrs in connected_edges:
                        if weight_key in edge_attrs:
                            temp_sum += edge_attrs[weight_key]
                    if temp_sum > 0:
                        total_weights.append(temp_sum)

            if total_weights:
                # Sort for percentile lookups
                weight_distributions[weight_key] = sorted(total_weights)
            else:
                weight_distributions[weight_key] = [0.0]

        for node in self.G.nodes():
            node_attrs = self.G.nodes[node]
            if node_attrs.get("type") == "odlocation":
                # get all edges connected to this odlocation
                connected_edges = self.G.edges(node, data=True)
                poi_weight_sums = defaultdict(lambda: defaultdict(float))
                poi_time_weight_sums = defaultdict(lambda: defaultdict(float))
                for u, v, edge_attrs in connected_edges:
                    # Static Weights
                    if (
                        "initial_weight" in edge_attrs
                        and "initial_x_decay_weight" in edge_attrs
                        and "initial_x_decay_x_tfidf_weight" in edge_attrs
                    ):
                        initial_weight = edge_attrs["initial_weight"]
                        initial_x_decay_weight = edge_attrs["initial_x_decay_weight"]
                        initial_x_decay_x_tfidf_weight = edge_attrs[
                            "initial_x_decay_x_tfidf_weight"
                        ]

                        # Determine if the connected node is a POI
                        other_node = v if u == node else u
                        other_node_attrs = self.G.nodes[other_node]
                        if other_node_attrs.get("type") == "poi":
                            category = other_node_attrs.get(
                                "category", other_node_attrs.get("name", "unknown")
                            )
                            poi_weight_sums["initial_weight"][
                                category
                            ] += initial_weight
                            poi_weight_sums["initial_x_decay_weight"][
                                category
                            ] += initial_x_decay_weight
                            poi_weight_sums["initial_x_decay_x_tfidf_weight"][
                                category
                            ] += initial_x_decay_x_tfidf_weight
                    else:
                        self.logger.warning(
                            f"Edge ({u}, {v}) is missing expected weight attributes."
                        )
                    # Time-based Weights
                    if (
                        "initial_weight_time" in edge_attrs
                        and "initial_weight_time_x_decay_weight" in edge_attrs
                        and "initial_weight_time_x_decay_x_tfidf_weight" in edge_attrs
                    ):
                        initial_weight_time = edge_attrs["initial_weight_time"]
                        initial_weight_time_x_decay_weight = edge_attrs[
                            "initial_weight_time_x_decay_weight"
                        ]
                        initial_weight_time_x_decay_x_tfidf_weight = edge_attrs[
                            "initial_weight_time_x_decay_x_tfidf_weight"
                        ]

                        # Determine if the connected node is a POI
                        other_node = v if u == node else u
                        other_node_attrs = self.G.nodes[other_node]
                        if other_node_attrs.get("type") == "poi":
                            category = other_node_attrs.get(
                                "category", other_node_attrs.get("name", "unknown")
                            )

                            if "initial_weight_time" not in poi_time_weight_sums:
                                poi_time_weight_sums["initial_weight_time"] = {}
                            if (
                                "initial_weight_time_x_decay_weight"
                                not in poi_time_weight_sums
                            ):
                                poi_time_weight_sums[
                                    "initial_weight_time_x_decay_weight"
                                ] = {}
                            if (
                                "initial_weight_time_x_decay_x_tfidf_weight"
                                not in poi_time_weight_sums
                            ):
                                poi_time_weight_sums[
                                    "initial_weight_time_x_decay_x_tfidf_weight"
                                ] = {}

                            if (
                                category
                                not in poi_time_weight_sums["initial_weight_time"]
                            ):
                                poi_time_weight_sums["initial_weight_time"][
                                    category
                                ] = {}
                                for day in initial_weight_time.keys():
                                    poi_time_weight_sums["initial_weight_time"][
                                        category
                                    ][day] = {}
                                    for hour_idx in range(24):
                                        poi_time_weight_sums["initial_weight_time"][
                                            category
                                        ][day][hour_idx] = 0.0
                            if (
                                category
                                not in poi_time_weight_sums[
                                    "initial_weight_time_x_decay_weight"
                                ]
                            ):
                                poi_time_weight_sums[
                                    "initial_weight_time_x_decay_weight"
                                ][category] = {}
                                for day in initial_weight_time_x_decay_weight.keys():
                                    poi_time_weight_sums[
                                        "initial_weight_time_x_decay_weight"
                                    ][category][day] = {}
                                    for hour_idx in range(24):
                                        poi_time_weight_sums[
                                            "initial_weight_time_x_decay_weight"
                                        ][category][day][hour_idx] = 0.0
                            if (
                                category
                                not in poi_time_weight_sums[
                                    "initial_weight_time_x_decay_x_tfidf_weight"
                                ]
                            ):
                                poi_time_weight_sums[
                                    "initial_weight_time_x_decay_x_tfidf_weight"
                                ][category] = {}
                                for (
                                    day
                                ) in initial_weight_time_x_decay_x_tfidf_weight.keys():
                                    poi_time_weight_sums[
                                        "initial_weight_time_x_decay_x_tfidf_weight"
                                    ][category][day] = {}
                                    for hour_idx in range(24):
                                        poi_time_weight_sums[
                                            "initial_weight_time_x_decay_x_tfidf_weight"
                                        ][category][day][hour_idx] = 0.0
                            if initial_weight_time:
                                for day, hours in initial_weight_time.items():
                                    # Hours is always a list of 24 values
                                    for hour_idx, hour_val in enumerate(hours):
                                        poi_time_weight_sums["initial_weight_time"][
                                            category
                                        ][day][hour_idx] += hour_val

                            if initial_weight_time_x_decay_weight:
                                for (
                                    day,
                                    hours,
                                ) in initial_weight_time_x_decay_weight.items():
                                    for hour_idx, hour_val in enumerate(hours):
                                        poi_time_weight_sums[
                                            "initial_weight_time_x_decay_weight"
                                        ][category][day][hour_idx] += hour_val

                            if initial_weight_time_x_decay_x_tfidf_weight:
                                for (
                                    day,
                                    hours,
                                ) in initial_weight_time_x_decay_x_tfidf_weight.items():
                                    for hour_idx, hour_val in enumerate(hours):
                                        poi_time_weight_sums[
                                            "initial_weight_time_x_decay_x_tfidf_weight"
                                        ][category][day][hour_idx] += hour_val

                # Update node attributes with the calculated sums
                node_attrs["poi_weight_sums"] = dict(poi_weight_sums)

                # Add private as a new category in poi_weight_sums if private_score exists
                if "private_score" in node_attrs:
                    static_private_score = node_attrs["private_score"]["static"]
                    for key in [
                        "initial_weight",
                        "initial_x_decay_weight",
                        "initial_x_decay_x_tfidf_weight",
                    ]:
                        if key in node_attrs["poi_weight_sums"]:
                            # Calculate total POI weight (publicness indicator)
                            total_poi_weight = sum(
                                node_attrs["poi_weight_sums"][key].values()
                            )

                            # Scale private score based on percentile ranking in distribution
                            # Higher percentile = more POIs = more public = higher penalty
                            distribution = weight_distributions.get(key, [0.0])

                            # Find percentile rank (0-1) of this location's POI weight
                            rank = sum(1 for w in distribution if w < total_poi_weight)
                            percentile = (
                                rank / len(distribution)
                                if len(distribution) > 0
                                else 0.0
                            )

                            # Map percentile to penalty:
                            # 0-25th percentile: 0-20% penalty (few POIs = mostly private)
                            # 50th percentile: 40% penalty (average)
                            # 75th percentile: 60% penalty
                            # 90th percentile: 80% penalty
                            # 100th percentile: 90% penalty (many POIs = mostly public)
                            publicness_penalty = min(percentile * 0.9, 0.9)
                            scaled_private_score = static_private_score * (
                                1 - publicness_penalty
                            )

                            # Add scaled private score to dict
                            node_attrs["poi_weight_sums"][key][
                                "private"
                            ] = scaled_private_score

                # Scale values proportionally to sum to 1, with private capped at 0.5 if other POIs exist
                for key in [
                    "initial_weight",
                    "initial_x_decay_weight",
                    "initial_x_decay_x_tfidf_weight",
                ]:
                    if key in node_attrs["poi_weight_sums"]:
                        # First do L1 normalization
                        node_attrs["poi_weight_sums"][key] = self._normalize_l1(
                            node_attrs["poi_weight_sums"][key]
                        )

                        # If private score exists and other POIs exist, cap private at threshold
                        if "private" in node_attrs["poi_weight_sums"][key]:
                            num_categories = len(node_attrs["poi_weight_sums"][key])
                            if num_categories > 1:  # Other POIs exist besides private
                                private_val = node_attrs["poi_weight_sums"][key][
                                    "private"
                                ]
                                if private_val > self.private_cap_threshold:
                                    # Cap private at threshold
                                    node_attrs["poi_weight_sums"][key][
                                        "private"
                                    ] = self.private_cap_threshold

                                    # Re-normalize other categories to sum to remaining proportion
                                    other_categories = {
                                        k: v
                                        for k, v in node_attrs["poi_weight_sums"][
                                            key
                                        ].items()
                                        if k != "private"
                                    }
                                    normalized_others = self._normalize_l1(
                                        other_categories
                                    )

                                    # Scale to remaining proportion
                                    remaining = 1.0 - self.private_cap_threshold
                                    for cat in normalized_others:
                                        node_attrs["poi_weight_sums"][key][cat] = (
                                            normalized_others[cat] * remaining
                                        )

                                    self.logger.info(
                                        f"Node {node}: Capped private score for {key} from {private_val:.4f} to {self.private_cap_threshold}, adjusted other categories"
                                    )

                node_attrs["poi_time_weight_sums"] = dict(poi_time_weight_sums)

                # Add private as a new category in poi_time_weight_sums if private_score exists
                if (
                    "private_score" in node_attrs
                    and "time_dependent" in node_attrs["private_score"]
                ):
                    time_private_scores = node_attrs["private_score"]["time_dependent"]

                    for key in [
                        "initial_weight_time",
                        "initial_weight_time_x_decay_weight",
                        "initial_weight_time_x_decay_x_tfidf_weight",
                    ]:
                        if key not in node_attrs["poi_time_weight_sums"]:
                            node_attrs["poi_time_weight_sums"][key] = {}

                        # Initialize private category
                        node_attrs["poi_time_weight_sums"][key]["private"] = {}

                        # Add time-dependent private scores with hourly POI activity dampening
                        for day, hour_scores in time_private_scores.items():
                            node_attrs["poi_time_weight_sums"][key]["private"][day] = {}
                            for hour_idx, score in enumerate(hour_scores):
                                # Calculate total POI activity at this specific hour
                                total_hour_activity = 0.0
                                for category in node_attrs["poi_time_weight_sums"][
                                    key
                                ].keys():
                                    if (
                                        category != "private"
                                        and day
                                        in node_attrs["poi_time_weight_sums"][key][
                                            category
                                        ]
                                    ):
                                        total_hour_activity += node_attrs[
                                            "poi_time_weight_sums"
                                        ][key][category][day].get(hour_idx, 0.0)

                                # Scale private score based on percentile ranking
                                # Map weight type keys to their static equivalents
                                if key == "initial_weight_time":
                                    distribution = weight_distributions.get(
                                        "initial_weight", [0.0]
                                    )
                                elif key == "initial_weight_time_x_decay_weight":
                                    distribution = weight_distributions.get(
                                        "initial_x_decay_weight", [0.0]
                                    )
                                else:  # initial_weight_time_x_decay_x_tfidf_weight
                                    distribution = weight_distributions.get(
                                        "initial_x_decay_x_tfidf_weight", [0.0]
                                    )

                                # Find percentile rank of this hour's POI activity
                                rank = sum(
                                    1 for w in distribution if w < total_hour_activity
                                )
                                percentile = (
                                    rank / len(distribution)
                                    if len(distribution) > 0
                                    else 0.0
                                )

                                # Map percentile to penalty (same as static)
                                publicness_penalty = min(percentile * 0.9, 0.9)
                                scaled_hour_score = score * (1 - publicness_penalty)

                                node_attrs["poi_time_weight_sums"][key]["private"][day][
                                    hour_idx
                                ] = scaled_hour_score

                # Scale time-based weights proportionally to sum to 1 per hour
                for key in [
                    "initial_weight_time",
                    "initial_weight_time_x_decay_weight",
                    "initial_weight_time_x_decay_x_tfidf_weight",
                ]:
                    # get all categories
                    if key in node_attrs["poi_time_weight_sums"]:
                        categories = list(
                            node_attrs["poi_time_weight_sums"][key].keys()
                        )
                        # Get all unique days across all categories
                        all_days = set()
                        for category in categories:
                            all_days.update(
                                node_attrs["poi_time_weight_sums"][key][category].keys()
                            )

                        for day in all_days:
                            for hour_idx in range(24):
                                # Collect weights for this hour across all categories
                                hour_dict = {}
                                for category in categories:
                                    if (
                                        day
                                        in node_attrs["poi_time_weight_sums"][key][
                                            category
                                        ]
                                    ):
                                        hour_dict[category] = node_attrs[
                                            "poi_time_weight_sums"
                                        ][key][category][day][hour_idx]
                                    else:
                                        hour_dict[category] = 0.0

                                # Normalize across categories for this specific hour
                                normalized_hour_dict = self._normalize_l1(hour_dict)

                                # If private exists and other POIs exist with non-zero values, cap private at threshold
                                if (
                                    "private" in normalized_hour_dict
                                    and len(normalized_hour_dict) > 1
                                ):
                                    # Check if any other category has a non-zero value at this hour
                                    other_categories = {
                                        k: v
                                        for k, v in normalized_hour_dict.items()
                                        if k != "private"
                                    }
                                    has_active_pois = any(
                                        v > 0 for v in other_categories.values()
                                    )

                                    if has_active_pois:
                                        private_val = normalized_hour_dict["private"]
                                        if private_val > self.private_cap_threshold:
                                            # Cap private at threshold
                                            normalized_hour_dict["private"] = (
                                                self.private_cap_threshold
                                            )

                                            # Re-normalize other categories to sum to remaining proportion
                                            normalized_others = self._normalize_l1(
                                                other_categories
                                            )

                                            # Scale to remaining proportion
                                            remaining = 1.0 - self.private_cap_threshold
                                            for cat in normalized_others:
                                                normalized_hour_dict[cat] = (
                                                    normalized_others[cat] * remaining
                                                )

                                # Write back the normalized values
                                for category in categories:
                                    if (
                                        day
                                        in node_attrs["poi_time_weight_sums"][key][
                                            category
                                        ]
                                    ):
                                        node_attrs["poi_time_weight_sums"][key][
                                            category
                                        ][day][hour_idx] = normalized_hour_dict[
                                            category
                                        ]

        self.logger.info("Node attributes updated based on connected edge weights.")

    def _normalize_l1(self, vec):
        total = sum(vec.values())
        if total == 0:
            return vec  # or return uniform distribution
        return {k: float(v) / total for k, v in vec.items()}
