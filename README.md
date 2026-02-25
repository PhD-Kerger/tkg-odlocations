# Inferring Micromobility Trip Context from Urban Points of Interest via a Temporal Knowledge Graph

We propose a temporal knowledge graph that directly connects micromobility locations with nearby POIs using routing-based walking distances, semantic rarity weighting (inverse document frequency), and opening-hour information. The method incrementally refines spatial relations through distance decay, category balancing, and temporal activation, resulting in interpretable, time-dependent, station-level context representations without relying on predefined spatial tessellations.

## Overview

This project implements a novel approach to understanding micromobility usage patterns by creating a temporal knowledge graph that directly links micromobility stations with nearby POIs while explicitly modeling:

- **Spatial distance** using routing-based walking distances
- **Semantic rarity** through inverse document frequency (IDF) weighting
- **Temporal availability** via opening-hour information
- **Optional land-use** for private trip purposes

![Map Example](docs/map.png)

Unlike traditional grid-based or hexagonal tessellation approaches, this framework preserves locality, avoids boundary effects, and provides interpretable, time-dependent embeddings for each station.

## Key Features

- **Data-source agnostic**: Works with any micromobility network providing O/D locations
- **Explainable by design**: Direct station-POI relationships without spatial aggregation
- **Temporal modeling**: Incorporates opening hours and daily activity patterns
- **Distance-aware**: Uses OSRM routing for realistic walking distances with exponential decay
- **Semantic balancing**: IDF weighting prevents dominance of frequent POI categories
- **Two output modes**:
  - **Graph mode**: Interactive visualization maps
  - **API mode**: JSON embeddings for machine learning features
- **Optional private trips**: Integrates land-use for non-POI destinations

## Requirements

### System Dependencies

- **Python**: 3.9+
- **Docker & Docker Compose**: For containerized deployment
- **PostgreSQL**: Database with micromobility and POI data
- **OSRM**: Routing engine (provided via Docker)

### Python Dependencies

See [requirements.txt](requirements.txt) for full list.

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:PhD-Kerger/tkg-odlocations.git
cd tkg-odlocations
```

### 2. Configure the Project

Copy the example configuration and adjust to your needs:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your settings (see [Configuration](#configuration) section).

### 3. Prepare OSRM Routing Data

To setup OSRM for walking routes follow the guide in our [Main Repository](https://github.com/PhD-Kerger/main).

### 4. Setup Database

Ensure your PostgreSQL database contains:

- Micromobility O/D locations (id, name, lat, lon)
- OSM POIs (osm_id, lat, lon, name, opening_hours, type)
- Optional: Land-use polygons

### 5. Build and Start Docker Services

First, build the Docker image:

```bash
docker build . -t tkg-odlocations
```

Then, start the services using Docker Compose:

```bash
docker-compose up -d
```

## Configuration

The `config.yaml` file controls all aspects of the pipeline:

### Mode Selection

```yaml
mode: "map" # "api" or "map"
```

- **map**: Generates interactive visualization map
- **api**: Processes single coordinates and returns JSON embeddings

### Database Connection

```yaml
database:
  user: "your_username"
  password: "your_password"
  host: "localhost"
  port: 5432
  dbname: "your_database"
```

### Processing Settings

```yaml
processing:
  enable_private_score: false # Enable land-use based private trip scoring
  private_cap_threshold: 0.7 # Max private score weight (0-1)

  location:
    city_name: "Mannheim"
    network_names: # List of micromobility networks
      - N-MH
      - N-MA
    year: 2023

  osm:
    enable_others_category: false # Include 'others' POI category

  data_preparation:
    max_air_distance_km: 0.5 # Initial BallTree radius
    max_walking_distance: 350 # OSRM walking distance threshold (meters)
    osrm_endpoint: "http://localhost:5000/route/v1/foot/"
```

## Usage

### Graph Mode (Visualization)

Generate an interactive map showing all stations with their POI embeddings:

```bash
python main.py
```

Output: An interactive Folium map saved to the workspace showing:

- Micromobility stations (colored by dominant category)
- Connected POIs
- Edge weights
- Temporal embeddings for each hour/weekday

### API Mode (Single Coordinate)

Process a single coordinate and get JSON embeddings:

```bash
python main.py 49.4875,8.4660,university
```

**Response Format**:

```json
{
  "latitude": 49.4875,
  "longitude": 8.466,
  "name": "University Station",
  "type": "odlocation",
  "landuse": {
    "work_percentage": 0.0386,
    "residential_percentage": 0.7162,
    "classification": "residential_dominant"
  },
  "static_embedding": {
    "food_beverage": {
      "overall": 0.0738,
      "day": 0.0625,
      "night": 0.0039
    },
    "education": {
      "overall": 0.1205,
      "day": 0.1102,
      "night": 0.0051
    }
  },
  "time_embedding": {
    "education": {
      "Monday": {
        "8": 0.65,
        "9": 0.72,
        "14": 0.68,
        "18": 0.45
      }
    }
  }
}
```
