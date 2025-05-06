## src/visualize/geoplot.py
"""
geoplot.py
----------

This visualization renders a 3-D plot of the data given the state
trajectory of a simulation, and the path of the property to render.

It generates an HTML file that contains code to render the plot
using Cesium Ion, and the GeoJSON file of data provided to the plot.

An example of its usage is as follows:

```py
from agent_torch.visualize import GeoPlot

# create a simulation
# ...

# create a visualizer
engine = GeoPlot(config, {
  cesium_token: "...",
  step_time: 3600,
  coordinates = "agents/consumers/coordinates",
  feature = "agents/consumers/money_spent",
})

# visualize in the runner-loop
for i in range(0, num_episodes):
  runner.step(num_steps_per_episode)
  engine.render(runner.state_trajectory)
```
"""

# Standard library imports for regex parsing and JSON serialization
import re  # regular expressions for splitting variable paths
import json  # encode Python objects as JSON

# Third-party libraries for data manipulation
import pandas as pd  # timestamp generation and time series handling
import numpy as np  # numeric arrays and conversions

# Utilities for templating and nested state access
from string import Template  # substitute placeholders in HTML template
from agent_torch.core.helpers import get_by_path  # access nested state via a path

# HTML template for Cesium-based visualization with placeholders:
#   $accessToken, $data, $startTime, $stopTime, $visualType
geoplot_template = """
<!doctype html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta
			name="viewport"
			content="width=device-width, initial-scale=1.0"
		/>
		<title>Cesium Time-Series Heatmap Visualization</title>
		<script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
		<link
			href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css"
			rel="stylesheet"
		/>
		<style>
			#cesiumContainer {
				width: 100%;
				height: 100%;
			}
		</style>
	</head>
	<body>
		<div id="cesiumContainer"></div>
		<script>
			// Ion access token placeholder
			Cesium.Ion.defaultAccessToken = '$accessToken'

			// Initialize Cesium viewer
			const viewer = new Cesium.Viewer('cesiumContainer')

			// Linearly interpolate between two colors based on factor
			function interpolateColor(color1, color2, factor) {
				const result = new Cesium.Color()
				result.red = color1.red + factor * (color2.red - color1.red)
				result.green = color1.green + factor * (color2.green - color1.green)
				result.blue = color1.blue + factor * (color2.blue - color1.blue)
				result.alpha = '$visualType' == 'size' ? 0.2 :
					color1.alpha + factor * (color2.alpha - color1.alpha)
				return result
			}

			// Map a value to a color gradient from blue to red
			function getColor(value, min, max) {
				const factor = (value - min) / (max - min)  
				return interpolateColor(
					Cesium.Color.BLUE,
					Cesium.Color.RED,
					factor
				)
			}

			// Map a value to a pixel size when visualType is 'size'
			function getPixelSize(value, min, max) {
				const factor = (value - min) / (max - min)
				return 100 * (1 + factor)
			}

			// Process GeoJSON features into time-series data grouped by entity ID
			function processTimeSeriesData(geoJsonData) {
				const timeSeriesMap = new Map()
				let minValue = Infinity  // track global min
				let maxValue = -Infinity // track global max

				geoJsonData.features.forEach((feature) => {
					const id = feature.properties.id
					const time = Cesium.JulianDate.fromIso8601(
						feature.properties.time
					)
					const value = feature.properties.value
					const coordinates = feature.geometry.coordinates

					// Initialize list for this entity if needed
					if (!timeSeriesMap.has(id)) {
						timeSeriesMap.set(id, [])
					}
					timeSeriesMap.get(id).push({ time, value, coordinates })

					// Update min/max for scaling
					minValue = Math.min(minValue, value)
					maxValue = Math.max(maxValue, value)
				})

				return { timeSeriesMap, minValue, maxValue }
			}

			// Create entities in Cesium for each ID with sampled position and styling
			function createTimeSeriesEntities(
				timeSeriesData,
				startTime,
				stopTime
			) {
				const dataSource = new Cesium.CustomDataSource(
					'AgentTorch Simulation'
				)

				// Iterate each entity's series
				for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
					const entity = new Cesium.Entity({
						id: id,
						availability: new Cesium.TimeIntervalCollection([
							new Cesium.TimeInterval({
								start: startTime,
								stop: stopTime,
							}),
						]),
						position: new Cesium.SampledPositionProperty(),  // dynamic path
						point: {
							pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
							color: new Cesium.SampledProperty(Cesium.Color),
						},
						properties: {
							value: new Cesium.SampledProperty(Number),
						},
					})

					// Add each sample (time, position, color, size)
					timeSeries.forEach(({ time, value, coordinates }) => {
						// Convert [lon, lat] to Cartesian3 position
						const position = Cesium.Cartesian3.fromDegrees(
							coordinates[0],
							coordinates[1]
						)
						entity.position.addSample(time, position)
						entity.properties.value.addSample(time, value)
						entity.point.color.addSample(
							time,
							getColor(
								value,
								timeSeriesData.minValue,
								timeSeriesData.maxValue
							)
						)

						// If visualType is size, sample pixel size as well
						if ('$visualType' == 'size') {
							entity.point.pixelSize.addSample(
								time,
								getPixelSize(
									value,
									timeSeriesData.minValue,
									timeSeriesData.maxValue
								)
							)
						}
					})

					dataSource.entities.add(entity)  # add to data source
				}

				return dataSource
			}

			// Load the list of GeoJSON time-series data
			const geoJsons = $data

			// Parse start/stop times for the simulation clock
			const start = Cesium.JulianDate.fromIso8601('$startTime')
			const stop = Cesium.JulianDate.fromIso8601('$stopTime')

			// Configure viewer clock playback range and speed
			viewer.clock.startTime = start.clone()
			viewer.clock.stopTime = stop.clone()
			viewer.clock.currentTime = start.clone()
			viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP
			viewer.clock.multiplier = 3600  # simulate 1 hour per second

			viewer.timeline.zoomTo(start, stop)

			// Render each GeoJSON as a Cesium data source
			for (const geoJsonData of geoJsons) {
				const timeSeriesData = processTimeSeriesData(geoJsonData)
				const dataSource = createTimeSeriesEntities(
					timeSeriesData,
					start,
					stop
				)
				viewer.dataSources.add(dataSource)
				viewer.zoomTo(dataSource)
			}
		</script>
	</body>
</html>
"""

def read_var(state, var):
    """Retrieve a nested variable from a state dict given a slash-separated path."""
    return get_by_path(state, re.split("/", var))


class GeoPlot:
    """Convert simulation state trajectories into Cesium visualizations."""

    def __init__(self, config, options):
        # Store simulation config and visualization settings
        self.config = config
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],  # Ion access token
            options["step_time"],     # seconds between steps
            options["coordinates"],   # path to position in state dict
            options["feature"],       # path to property values in state dict
            options["visualization_type"],  # 'color' or 'size'
        )

    def render(self, state_trajectory):
        """Generate GeoJSON and HTML files from a list of state dicts."""
        coords, values = [], []
        name = self.config["simulation_metadata"]["name"]  # base filename
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # Extract coordinates and feature values at each step
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]

            # Read the nested coordinate list and flatten to Python list
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            # Read feature array, flatten, and store
            values.append(
                np.array(read_var(final_state, self.entity_property))
                .flatten()
                .tolist()
            )

        # Generate timestamps for each step in the simulation
        start_time = pd.Timestamp.utcnow()  # use UTC now as timeline start
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        geojsons = []  # will hold FeatureCollections for each entity
        # Build GeoJSON features per coordinate index
        for i, coord in enumerate(coords):
            features = []
            for time, value_list in zip(timestamps, values):
                # Create a GeoJSON Feature with geometry and time/value props
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            # Cesium expects [lon, lat]
                            "coordinates": [coord[1], coord[0]],
                        },
                        "properties": {
                            "value": value_list[i],
                            "time": time.isoformat(),
                        },
                    }
                )
            # Wrap features into a FeatureCollection
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write GeoJSON data to file for later loading in HTML
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Fill HTML template placeholders and write out the final viewer page
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,
                        "startTime": timestamps[0].isoformat(),
                        "stopTime": timestamps[-1].isoformat(),
                        "data": json.dumps(geojsons),
                        "visualType": self.visualization_type,
                    }
                )
            )
