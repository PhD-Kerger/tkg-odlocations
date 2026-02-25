import base64
import io
import os
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .logger import Logger


class Visualizer:
    def __init__(self, G):
        self.G = G

        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def plot_graph_map(self):
        """
        Plots a map visualizing the graph with nodes and edges.
        Nodes are color-coded based on their type (odlocation, poi).
        Edges are drawn with colors and weights based on their attributes.
        Saves the map as an HTML file.
        """
        # Calculate center of the map based on node coordinates
        lats = []
        lons = []
        for node, attrs in self.G.nodes(data=True):
            if "lat" in attrs and "lon" in attrs:
                lats.append(attrs["lat"])
                lons.append(attrs["lon"])

        if lats and lons:
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            self.logger.info(
                f"Calculated map center: {center_lat:.3f}, {center_lon:.3f}"
            )

            # Create the map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Create feature groups for different elements
            edge_group = folium.FeatureGroup(name="Edges")
            odlocation_group = folium.FeatureGroup(name="O/D-Locations")
            poi_group = folium.FeatureGroup(name="POIs")

            # Define clusters for the markers
            odlocation_cluster = MarkerCluster(name="O/D-Locations Cluster").add_to(
                odlocation_group
            )
            poi_cluster = MarkerCluster(name="POI Cluster").add_to(poi_group)

            # Create a dictionary for quick lookup of node coordinates
            node_coords = {}
            node_attrs = {}
            for node, attrs in self.G.nodes(data=True):
                if "lat" in attrs and "lon" in attrs:
                    node_coords[node] = (attrs["lat"], attrs["lon"])
                    node_attrs[node] = attrs

            # Add edges
            self.logger.info("Adding edges...")
            added_edges = 0

            for source, target, edge in self.G.edges(data=True):
                # Check if both nodes have coordinates
                if source in node_coords and target in node_coords:
                    source_coords = node_coords[source]
                    target_coords = node_coords[target]
                    # Configure edge appearance based on edge type
                    edge_type = edge["edge_type"]
                    if "poi" in edge_type.lower():
                        color = "green"
                    else:
                        color = "blue"

                    # Determine line weight based on weight attribute
                    initial_x_decay_x_tfidf_weight = float(
                        edge["initial_x_decay_x_tfidf_weight"]
                    )
                    line_weight = max(0.5, min(5, initial_x_decay_x_tfidf_weight * 5))

                    # Add line without popup or tooltip
                    folium.PolyLine(
                        locations=[source_coords, target_coords],
                        color=color,
                        weight=line_weight,
                        opacity=0.6,
                    ).add_to(edge_group)
                    added_edges += 1

            self.logger.info(f"Added {added_edges} edges to the map")

            # Add nodes markers
            self.logger.info("Adding node markers...")

            # Collect locations for heatmaps
            odlocation_locations = []
            poi_locations = []

            for node, attrs in node_attrs.items():
                node_type = attrs.get("type", "unknown")
                name = attrs.get("name", str(node))
                coords = node_coords[node]

                # Collect locations for heatmaps
                if node_type == "odlocation":
                    odlocation_locations.append([coords[0], coords[1]])
                elif node_type == "poi":
                    poi_locations.append([coords[0], coords[1]])

                # Choose color and icon based on node type
                if node_type == "odlocation":
                    color = "blue"
                    icon = "random"
                    target_cluster = odlocation_cluster
                elif node_type == "poi":
                    color = "green"
                    icon = "info-sign"
                    target_cluster = poi_cluster
                else:
                    continue  # Skip unknown node types

                # Add enhanced info box and charts for odlocations
                if node_type == "odlocation":
                    # Generate time-based chart
                    final_chart = self._create_time_based_type_chart(
                        node, chart_type="timexdecayxtfidf"
                    )

                    # Create info box with station details
                    info_box = self._create_station_info_box(node, attrs)

                    # Create static embedding display
                    embedding_box = self._create_embedding_box(attrs)

                    # Create table of connected POIs
                    poi_table = self._create_connected_pois_table(node)

                    # Assemble the popup content with scrollable wrapper (plot before table)
                    station_name = attrs.get("name", "Unknown Station")
                    h1_heading = f'<h1 style="margin: 0 0 15px 0; color: #2c5aa0; font-size: 20px; border-bottom: 2px solid #4682b4; padding-bottom: 8px;">📍 {station_name}</h1>'
                    popup_content = (
                        h1_heading
                        + info_box
                        + "<br>"
                        + embedding_box
                        + "<br>"
                        + final_chart
                        + "<br>"
                        + poi_table
                    )
                    popup_text = f'<div style="max-height: 600px; overflow-y: auto; overflow-x: hidden;">{popup_content}</div>'

                # Add enhanced info box and table for POIs
                elif node_type == "poi":

                    # Create info box with POI details
                    poi_info_box = self._create_poi_info_box(node, attrs)

                    # Create table of connected O/D-locations
                    stations_table = self._create_connected_stations_table(node)

                    # Assemble the popup content
                    poi_name = attrs.get("name", "Unknown POI")
                    h1_heading = f'<h1 style="margin: 0 0 15px 0; color: #2c5aa0; font-size: 20px; border-bottom: 2px solid #4682b4; padding-bottom: 8px;">📍 {poi_name}</h1>'
                    popup_content = h1_heading + poi_info_box + "<br>" + stations_table
                    popup_text = f'<div style="max-height: 600px; overflow-y: auto; overflow-x: hidden;">{popup_content}</div>'

                # Create tooltip with additional information for different node types
                tooltip = name
                if node_type == "poi" and "category" in attrs:
                    tooltip += f" ({attrs['category']})"

                # Add markers
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_text, max_width=800),
                    tooltip=tooltip,
                    icon=folium.Icon(color=color, icon=icon),
                ).add_to(target_cluster)

            # Add heatmaps
            if odlocation_locations:
                HeatMap(
                    odlocation_locations,
                    name="O/D-Locations Heatmap",
                    show=False,
                    radius=15,
                    min_opacity=0.5,
                    gradient={0.2: "blue", 0.5: "blue", 0.8: "blue"},
                ).add_to(m)

            if poi_locations:
                HeatMap(
                    poi_locations,
                    name="POIs Heatmap",
                    show=False,
                    radius=15,
                    min_opacity=0.5,
                    gradient={0.2: "green", 0.5: "green", 0.8: "green"},
                ).add_to(m)

            # Add all feature groups to the map
            edge_group.add_to(m)
            odlocation_group.add_to(m)
            poi_group.add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            # Save the map as HTML
            if not os.path.exists(Path("data") / "maps"):
                os.makedirs(Path("data") / "maps")

            m.save(Path("data") / "maps" / "graph_visualization_map.html")
            self.logger.info("Map saved to data/maps/graph_visualization_map.html")
        else:
            self.logger.error(
                "No valid node coordinates found. Map could not be created."
            )

    def _create_static_type_chart(self, node, width=10, height=6):
        """
        Creates a bar chart for the POI types of an odlocation

        Args:
            odlocation_id: ID of the odlocation
            width, height: Size of the chart
        Returns:
            HTML code with embedded image of the chart
        """

        # Define cache directory and file path
        cache_dir = Path("data") / "chart_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"static_chart_{node}.png"

        # Check if cached image exists
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                image_png = f.read()
            encoded = base64.b64encode(image_png).decode("utf-8")
            html = f'<img src="data:image/png;base64,{encoded}">'
            return html
        # Extract data for this odlocation
        count_data = (
            self.G.nodes[node].get("poi_weight_sums", {}).get("initial_weight", {})
        )
        tfidf_data = (
            self.G.nodes[node]
            .get("poi_weight_sums", {})
            .get("initial_x_decay_x_tfidf_weight", {})
        )
        decay_data = (
            self.G.nodes[node]
            .get("poi_weight_sums", {})
            .get("initial_x_decay_weight", {})
        )

        # Use all POI types that have at least one value (Count, IDF, or Decay)
        poi_types = set()
        poi_types.update(count_data.keys())
        poi_types.update(tfidf_data.keys())
        poi_types.update(decay_data.keys())

        # Sort the POI types by count values (descending)
        poi_types = sorted(poi_types, key=lambda x: count_data.get(x, 0), reverse=True)

        if not poi_types:
            return "<p>No POIs found for this O/D-Location</p>"
        # Extract values for each type in the correct order
        count_values = [count_data.get(t, 0) for t in poi_types]
        tfidf_values = [tfidf_data.get(t, 0) for t in poi_types]
        decay_values = [decay_data.get(t, 0) for t in poi_types]

        # Create the chart
        fig, ax = plt.subplots(figsize=(width, height))

        x = np.arange(len(poi_types))
        bar_width = 0.25

        # Plot the bars for each aggregation method
        bar1 = ax.bar(
            x - bar_width, count_values, bar_width, color="skyblue", label="Count"
        )
        bar2 = ax.bar(
            x,
            decay_values,
            bar_width,
            color="green",
            alpha=0.7,
            label="Decay (Gaussian)",
        )
        bar3 = ax.bar(
            x + bar_width,
            tfidf_values,
            bar_width,
            color="orange",
            alpha=0.7,
            label="IDF",
        )

        # Labels and layout
        ax.set_title(f"Chart for O/D-Location {node}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(poi_types, rotation=45, ha="right", fontsize=9)

        # Set a meaningful y-axis label
        ax.set_ylabel("Values", fontsize=12)

        # Add labels for all values above the bars
        def add_value_labels(bars):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:  # Show only values > 0
                    value_text = f"{height:.2f}" if height < 10 else f"{height:.1f}"
                    if height == int(height):  # Show integers without decimal places
                        value_text = f"{int(height)}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.05,
                        value_text,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Add labels for all three bar types
        add_value_labels(bar1)
        add_value_labels(bar2)
        add_value_labels(bar3)

        # Add a legend
        ax.legend(fontsize=10)

        plt.tight_layout()

        # Save to buffer and cache file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.savefig(cache_file, format="png", dpi=100)  # Save to disk
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode as Base64 and return the HTML code
        encoded = base64.b64encode(image_png).decode("utf-8")
        html = f'<img src="data:image/png;base64,{encoded}">'
        plt.close(fig)  # Close the chart to free up memory
        return html

    def _create_time_based_type_chart(
        self, node, width=7, height=4, chart_type="inital"
    ):
        """
        Creates a bar chart for the time-based POI weights of an O/D-Location

        Args:
            odlocation_id: ID of the O/D-Location
            width, height: Size of the chart
        Returns:
            HTML code with embedded image of the chart
        """

        # Define fixed colors for categories
        category_colors = {
            "food_beverage": "#E74C3C",  # Strong Red
            "transportation": "#3498DB",  # Strong Blue
            "education": "#2ECC71",  # Strong Green
            "services_shopping": "#F39C12",  # Strong Orange
            "entertainment": "#9B59B6",  # Purple
            "private": "#F30CCD",  # Cyan/Turquoise
        }

        # Define formatted category names
        category_labels = {
            "food_beverage": "Food & Beverage",
            "transportation": "Transportation",
            "education": "Education",
            "services_shopping": "Services & Shopping",
            "entertainment": "Entertainment",
            "private": "Private",
        }

        # Define cache directory and file path
        cache_dir = Path("data") / "chart_cache" / chart_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"time_based_chart_{node}.png"

        # Check if cached image exists
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                image_png = f.read()
            encoded = base64.b64encode(image_png).decode("utf-8")

            # Create HTML legend (even when using cache)
            legend_html = '<div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">'
            for category in [
                "food_beverage",
                "transportation",
                "education",
                "services_shopping",
                "entertainment",
                "private",
            ]:
                if category in category_colors:
                    color = category_colors[category]
                    label = category_labels[category]
                    legend_html += f'<div style="display: flex; align-items: center; gap: 5px;"><div style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px;"></div><span style="font-size: 11px;">{label}</span></div>'
            legend_html += "</div>"

            # Wrap in fancy box (same as non-cached version)
            html = f"""
            <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
                <h4 style="margin-top: 0; color: #2c5aa0;">📊 Time-Based POI Weights</h4>
                {legend_html}
                <img src="data:image/png;base64,{encoded}">
            </div>
            """
            return html
        # Extract data for this odlocation
        time_based_data = self.G.nodes[node].get("poi_time_weight_sums", {})
        if chart_type == "inital":
            time_data = time_based_data.get("initial_weight_time")
        elif chart_type == "timexdecay":
            time_data = time_based_data.get("initial_weight_time_x_decay_weight")
        elif chart_type == "timexdecayxtfidf":
            time_data = time_based_data.get(
                "initial_weight_time_x_decay_x_tfidf_weight"
            )

        if time_data is None:
            return "<p>No time-based POI data found for this O/D-Location</p>"

        # create a line plot
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
        day_abbreviations = {
            "monday": "Mon",
            "tuesday": "Tue",
            "wednesday": "Wed",
            "thursday": "Thu",
            "friday": "Fri",
            "saturday": "Sat",
            "sunday": "Sun",
        }

        all_days = set()
        for category_data in time_data.values():
            all_days.update(category_data.keys())
        all_days = sorted(all_days, key=lambda day: day_order.index(day))
        # Create x-axis positions (24 hours per day)
        total_hours = len(all_days) * 24
        x_positions = list(range(total_hours))

        # Plot one line per category with fixed colors
        for category, day_data in time_data.items():
            # Concatenate all hour values across days
            category_values = []
            for day in all_days:
                if day in day_data:
                    category_values.extend(day_data[day].values())
                else:
                    category_values.extend([0] * 24)  # Fill with zeros if day missing

            # Get color and label for this category
            color = category_colors.get(
                category, "#808080"
            )  # Default gray if not found
            label = category_labels.get(category, category)

            ax.plot(
                x_positions,
                category_values,
                label=label,
                color=color,
                marker="o",
                markersize=2,
                linewidth=1.5,
            )

        # Set x-axis labels to show day boundaries with abbreviated day names
        day_boundaries = [i * 24 for i in range(len(all_days))]
        ax.set_xticks(day_boundaries)
        ax.set_xticklabels([day_abbreviations[day] for day in all_days], rotation=0)

        # Add vertical lines at day boundaries
        for boundary in day_boundaries[1:]:
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.3)

        plt.ylabel("Weight", fontsize=9)
        plt.tight_layout()

        # Save to buffer and cache file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.savefig(
            cache_file, format="png", dpi=100, bbox_inches="tight"
        )  # Save to disk
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode as Base64
        encoded = base64.b64encode(image_png).decode("utf-8")

        # Create HTML legend
        legend_html = '<div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">'
        for category in [
            "food_beverage",
            "transportation",
            "education",
            "services_shopping",
            "entertainment",
            "private",
        ]:
            if category in category_colors:
                color = category_colors[category]
                label = category_labels[category]
                legend_html += f'<div style="display: flex; align-items: center; gap: 5px;"><div style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px;"></div><span style="font-size: 11px;">{label}</span></div>'
        legend_html += "</div>"

        # Combine legend and image in a fancy box
        html = f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="margin-top: 0; color: #2c5aa0;">📊 Time-Based POI Weights</h4>
            {legend_html}
            <img src="data:image/png;base64,{encoded}">
        </div>
        """
        plt.close(fig)  # Close the chart to free up memory

        return html

    def _create_station_info_box(self, node, attrs):
        """
        Creates an HTML info box with important station details

        Args:
            node: Node ID
            attrs: Node attributes dictionary
        Returns:
            HTML string with station information
        """
        name = attrs.get("name", "Unknown")
        lat = attrs.get("lat", "N/A")
        lon = attrs.get("lon", "N/A")

        # Count connected POIs by getting edges
        connected_pois = []
        for _, target, edge_data in self.G.edges(node, data=True):
            if self.G.nodes[target].get("type") == "poi":
                connected_pois.append(target)

        total_pois = len(connected_pois)

        # Get unique POI categories
        poi_categories = set()
        for poi_id in connected_pois:
            category = self.G.nodes[poi_id].get("category", "Unknown")
            poi_categories.add(category)

        info_box = f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="margin-top: 0; color: #2c5aa0;">📍 General Information</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>Type:</b></td><td>O/D-Location</td></tr>
                <tr><td><b>ID:</b></td><td>{node}</td></tr>
                <tr><td><b>Coordinates:</b></td><td>{lat:.3f}, {lon:.3f}</td></tr>
                <tr><td><b>Connected POIs:</b></td><td>{total_pois}</td></tr>
                <tr><td><b>POI Categories:</b></td><td>{len(poi_categories)}</td></tr>
            </table>
        </div>
        """
        return info_box

    def _create_embedding_box(self, attrs):
        """
        Creates an HTML box displaying the static embedding values

        Args:
            attrs: Node attributes dictionary
        Returns:
            HTML string with embedding information
        """
        # Define formatted category names
        category_labels = {
            "food_beverage": "Food & Beverage",
            "transportation": "Transportation",
            "education": "Education",
            "services_shopping": "Services & Shopping",
            "entertainment": "Entertainment",
            "private": "Private",
        }

        # Calculate static embedding from time-based POI weights
        poi_time_weight_sums = attrs.get("poi_time_weight_sums", {})
        time_data = poi_time_weight_sums.get(
            "initial_weight_time_x_decay_x_tfidf_weight", {}
        )

        if not time_data:
            return ""

        # Calculate per-category embeddings
        category_embeddings = {}

        for category, days in time_data.items():
            day_sum = 0.0
            night_sum = 0.0
            overall_sum = 0.0
            day_count = 0
            night_count = 0
            overall_count = 0

            for day, hours in days.items():
                for hour, value in hours.items():
                    hour_int = int(hour)
                    overall_sum += value
                    overall_count += 1

                    # Day is 6am (6) to 9pm (21)
                    if 6 <= hour_int <= 21:
                        day_sum += value
                        day_count += 1
                    # Night is before 6am or after 9pm
                    else:
                        night_sum += value
                        night_count += 1

            # Calculate averages
            day_avg = round(day_sum / day_count, 4) if day_count > 0 else 0.0
            night_avg = round(night_sum / night_count, 4) if night_count > 0 else 0.0
            overall_avg = (
                round(overall_sum / overall_count, 4) if overall_count > 0 else 0.0
            )

            category_embeddings[category] = {
                "day": day_avg,
                "night": night_avg,
                "overall": overall_avg,
            }

        # Build HTML table
        embedding_box = """
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="margin-top: 0; color: #2c5aa0;">🔢 Static Embedding (Averages)</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                <thead style="background-color: #4682b4; color: white;">
                    <tr>
                        <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">Category</th>
                        <th style="padding: 4px; text-align: right; border: 1px solid #ddd;">Day Avg<br><small style="font-size: 9px;">(6am-9pm)</small></th>
                        <th style="padding: 4px; text-align: right; border: 1px solid #ddd;">Night Avg<br><small style="font-size: 9px;">(9pm-6am)</small></th>
                        <th style="padding: 4px; text-align: right; border: 1px solid #ddd;">Overall Avg<br><small style="font-size: 9px;">(24h)</small></th>
                    </tr>
                </thead>
                <tbody>
        """

        for idx, (category, values) in enumerate(category_embeddings.items()):
            row_color = "#ffffff" if idx % 2 == 0 else "#f2f2f2"
            formatted_category = category_labels.get(category, category)
            embedding_box += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 3px; border: 1px solid #ddd;"><b>{formatted_category}</b></td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd;">{values['day']}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd;">{values['night']}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd;">{values['overall']}</td>
                    </tr>
            """

        embedding_box += """
                </tbody>
            </table>
        </div>
        """
        return embedding_box

    def _create_connected_pois_table(self, node):
        """
        Creates an HTML table showing all connected POIs with their details

        Args:
            node: O/D-Location node ID
        Returns:
            HTML string with POI table
        """
        # Define formatted category names (same as in chart)
        category_labels = {
            "food_beverage": "Food & Beverage",
            "transportation": "Transportation",
            "education": "Education",
            "services_shopping": "Services & Shopping",
            "entertainment": "Entertainment",
            "private": "Private",
        }

        entity_name_labels = {
            "restaurant": "Restaurant",
            "fastfood": "Fast Food",
            "cafe": "Café",
            "bakery": "Bakery",
            "bar": "Bar",
            "tram_stop": "Tram Stop",
            "station": "Station",
            "school": "School",
            "college": "College",
            "university": "University",
            "library": "Library",
            "clothes_shop": "Clothes Shop",
            "supermarket": "Supermarket",
            "marketplace": "Marketplace",
            "department_store_shop": "Department Store",
            "bank": "Bank",
            "townhall": "Town Hall",
            "place_of_worship": "Place of Worship",
            "dentist": "Dentist",
            "pharmacy": "Pharmacy",
            "chemist_shop": "Chemist Shop",
            "theatre": "Theatre",
            "cinema": "Cinema",
        }

        # Collect all connected POIs with their data
        poi_data = []
        for _, target, edge_data in self.G.edges(node, data=True):
            if self.G.nodes[target].get("type") == "poi":
                poi_name = self.G.nodes[target].get("name", "Unknown")
                poi_category = self.G.nodes[target].get("category", "Unknown")
                entity_name = self.G.nodes[target].get("entity_name", "Unknown")
                # Format the category name and entity name
                formatted_category = category_labels.get(poi_category, poi_category)
                formatted_entity_name = entity_name_labels.get(entity_name, entity_name)
                distance = edge_data.get("distance", 0)
                weight = edge_data.get("initial_x_decay_x_tfidf_weight", 0)

                poi_data.append(
                    {
                        "name": poi_name,
                        "category": formatted_category,
                        "entity_name": formatted_entity_name,
                        "distance": distance,
                        "weight": weight,
                    }
                )

        # Sort by weight (descending)
        poi_data.sort(key=lambda x: x["weight"], reverse=True)

        # Create HTML table
        table_html = """
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="color: #2c5aa0; margin-top: 0;">🔗 Connected POIs</h4>
            <div style="max-height: 200px; overflow-y: auto; overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 10px;">
                    <thead style="background-color: #4682b4; color: white; position: sticky; top: 0;">
                        <tr>
                            <th style="padding: 4px; text-align: left; border: 1px solid #ddd; max-width: 150px;">POI Name</th>
                            <th style="padding: 4px; text-align: left; border: 1px solid #ddd; max-width: 100px;">Category</th>
                            <th style="padding: 4px; text-align: left; border: 1px solid #ddd; max-width: 100px;">Entity Type</th>
                            <th style="padding: 4px; text-align: right; border: 1px solid #ddd; width: 70px;">Dist (m)</th>
                            <th style="padding: 4px; text-align: right; border: 1px solid #ddd; width: 60px;">Weight</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for idx, poi in enumerate(poi_data):
            row_color = "#ffffff" if idx % 2 == 0 else "#f2f2f2"
            table_html += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 3px; border: 1px solid #ddd; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{poi['name']}</td>
                        <td style="padding: 3px; border: 1px solid #ddd; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{poi['category']}</td>
                        <td style="padding: 3px; border: 1px solid #ddd; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{poi['entity_name']}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd; width: 70px;">{poi['distance']:.0f}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd; width: 60px;">{poi['weight']:.4f}</td>
                    </tr>
            """

        table_html += """
                </tbody>
            </table>
            </div>
        </div>
        """

        if not poi_data:
            return "<p><i>No connected POIs found</i></p>"

        return table_html

    def _create_poi_info_box(self, node, attrs):
        """
        Creates an HTML info box with important POI details

        Args:
            node: Node ID
            attrs: Node attributes dictionary
        Returns:
            HTML string with POI information
        """
        # Define formatted category names
        category_labels = {
            "food_beverage": "Food & Beverage",
            "transportation": "Transportation",
            "education": "Education",
            "services_shopping": "Services & Shopping",
            "entertainment": "Entertainment",
            "private": "Private",
        }

        entity_name_labels = {
            "restaurant": "Restaurant",
            "fastfood": "Fast Food",
            "cafe": "Café",
            "bakery": "Bakery",
            "bar": "Bar",
            "tram_stop": "Tram Stop",
            "station": "Station",
            "school": "School",
            "college": "College",
            "university": "University",
            "library": "Library",
            "clothes_shop": "Clothes Shop",
            "supermarket": "Supermarket",
            "marketplace": "Marketplace",
            "department_store_shop": "Department Store",
            "bank": "Bank",
            "townhall": "Town Hall",
            "place_of_worship": "Place of Worship",
            "dentist": "Dentist",
            "pharmacy": "Pharmacy",
            "chemist_shop": "Chemist Shop",
            "theatre": "Theatre",
            "cinema": "Cinema",
        }

        name = attrs.get("name", "Unknown")
        poi_id = attrs.get("poi_id", "N/A")
        category = attrs.get("category", "Unknown")
        entity_name = attrs.get("entity_name", "Unknown")
        # Format the category and entity name
        formatted_category = category_labels.get(category, category)
        formatted_entity_name = entity_name_labels.get(entity_name, entity_name)
        lat = attrs.get("lat", "N/A")
        lon = attrs.get("lon", "N/A")

        # Count connected O/D-locations
        connected_stations = []
        for u, v, edge_data in self.G.edges(node, data=True):
            # Determine which node is the station (not the POI)
            other_node = v if u == node else u
            if self.G.nodes[other_node].get("type") == "odlocation":
                connected_stations.append(other_node)

        total_stations = len(connected_stations)

        # Check for opening hours
        has_opening_hours = "opening_hours" in attrs and attrs["opening_hours"]
        opening_hours_status = "✓ Available" if has_opening_hours else "✗ Not Available"

        info_box = f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="margin-top: 0; color: #2c5aa0;">ℹ️ POI Information</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>Type:</b></td><td>Point of Interest</td></tr>
                <tr><td><b>Category:</b></td><td>{formatted_category}</td></tr>
                <tr><td><b>Entity Type:</b></td><td>{formatted_entity_name}</td></tr>
                <tr><td><b>POI ID:</b></td><td>{poi_id}</td></tr>
                <tr><td><b>Coordinates:</b></td><td>{lat:.3f}, {lon:.3f}</td></tr>
                <tr><td><b>Connected O/D-Locations:</b></td><td>{total_stations}</td></tr>
                <tr><td><b>Opening Hours:</b></td><td>{opening_hours_status}</td></tr>
            </table>
        </div>
        """
        return info_box

    def _create_connected_stations_table(self, node):
        """
        Creates an HTML table showing all O/D-locations connected to this POI

        Args:
            node: POI node ID
        Returns:
            HTML string with connected stations table
        """
        # Collect all connected stations with their data
        station_data = []
        for u, v, edge_data in self.G.edges(node, data=True):
            # Determine which node is the station (not the POI)
            other_node = v if u == node else u
            if self.G.nodes[other_node].get("type") == "odlocation":
                station_name = self.G.nodes[other_node].get("name", "Unknown")
                station_network = self.G.nodes[other_node].get("network", "Unknown")
                distance = edge_data.get("distance", 0)
                weight = edge_data.get("initial_x_decay_x_tfidf_weight", 0)

                station_data.append(
                    {
                        "name": station_name,
                        "network": station_network,
                        "distance": distance,
                        "weight": weight,
                    }
                )

        # Sort by weight (descending)
        station_data.sort(key=lambda x: x["weight"], reverse=True)

        # Create HTML table
        table_html = """
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4;">
            <h4 style="color: #2c5aa0; margin-top: 0;">🚉 Connected O/D-Locations</h4>
            <div style="max-height: 200px; overflow-y: auto; overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 10px;">
                    <thead style="background-color: #4682b4; color: white; position: sticky; top: 0;">
                        <tr>
                            <th style="padding: 4px; text-align: left; border: 1px solid #ddd; max-width: 150px;">Station Name</th>
                            <th style="padding: 4px; text-align: left; border: 1px solid #ddd; max-width: 100px;">Network</th>
                            <th style="padding: 4px; text-align: right; border: 1px solid #ddd; width: 70px;">Dist (m)</th>
                            <th style="padding: 4px; text-align: right; border: 1px solid #ddd; width: 60px;">Weight</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for idx, station in enumerate(station_data):
            row_color = "#ffffff" if idx % 2 == 0 else "#f2f2f2"
            table_html += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 3px; border: 1px solid #ddd; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{station['name']}">{station['name']}</td>
                        <td style="padding: 3px; border: 1px solid #ddd; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{station['network']}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd; width: 70px;">{station['distance']:.0f}</td>
                        <td style="padding: 3px; text-align: right; border: 1px solid #ddd; width: 60px;">{station['weight']:.4f}</td>
                    </tr>
            """

        table_html += """
                </tbody>
            </table>
            </div>
        </div>
        """

        if not station_data:
            return "<p><i>No connected stations found</i></p>"

        return table_html

    def _get_node_name_from_graph(self, node_id):
        """
        Get the name of a node from the graph

        Args:
            node_id: ID of the node
        """
        if node_id in self.G.nodes:
            return self.G.nodes[node_id].get("name", "N/A")
