import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io
import xml.etree.ElementTree as ET
import re
from shapely.geometry import Point, Polygon, shape, mapping
import base64
import zipfile
from io import BytesIO
import json
import math
import difflib
import unicodedata

def extract_coordinates_from_kml(kml_content):
    """
    Extract polygon coordinates from KML content with improved error handling 
    and duplicate territory name handling
    """
    # Parse the KML content
    try:
        root = ET.fromstring(kml_content)
    except ET.ParseError as e:
        st.error(f"Error parsing KML file: {e}")
        return {}
    
    # Define the XML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Dictionary to store territory polygons
    territories = {}
    
    # Dictionary to track duplicate territory names
    duplicate_counts = {}
    
    # Find all Placemark elements
    placemarks = root.findall('.//kml:Placemark', ns)
    
    total_polygons = 0
    valid_polygons = 0
    duplicate_territories = 0
    st.info(f"Found {len(placemarks)} placemarks in KML file")
    
    # First, do a scan of all territory names to identify duplicates and prepare for debugging
    all_territory_names = []
    for placemark in placemarks:
        name_elem = placemark.find('kml:name', ns)
        if name_elem is not None:
            raw_name = name_elem.text
            all_territory_names.append(raw_name)
    
    # Find and report duplicates
    name_counts = {}
    for name in all_territory_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    duplicate_names = [name for name, count in name_counts.items() if count > 1]
    if duplicate_names:
        st.warning(f"Found {len(duplicate_names)} duplicate territory names in KML file:")
        for i, name in enumerate(duplicate_names[:10]):  # Show first 10 to avoid cluttering the UI
            st.warning(f"{i+1}. '{name}' appears {name_counts[name]} times")
        if len(duplicate_names) > 10:
            st.warning(f"...and {len(duplicate_names) - 10} more duplicate names")
    
    # Process each Placemark
    for placemark in placemarks:
        # Get the name of the territory
        name_elem = placemark.find('kml:name', ns)
        if name_elem is not None:
            original_name = name_elem.text.strip()
            territory_name = original_name
            
            # Check for duplicate name and make it unique if necessary
            if territory_name in territories:
                # Count this duplicate
                duplicate_counts[territory_name] = duplicate_counts.get(territory_name, 1) + 1
                counter = duplicate_counts[territory_name]
                
                # Create a unique name by adding a suffix
                unique_name = f"{territory_name} #{counter}"
                
                st.warning(f"Duplicate territory name detected: '{territory_name}'. Renamed to '{unique_name}'")
                territory_name = unique_name
                duplicate_territories += 1
            
            # Debug info to detect invisible characters
            clean_name = unicodedata.normalize('NFKC', territory_name)
            if clean_name != territory_name:
                st.warning(f"Normalized territory name: '{territory_name}' → '{clean_name}'")
                # Use the normalized name
                territory_name = clean_name
        else:
            continue  # Skip if no name found
        
        # Find polygon coordinates
        coordinates_elem = placemark.find('.//kml:coordinates', ns)
        if coordinates_elem is not None:
            total_polygons += 1
            
            try:
                # Extract and clean coordinate string
                coord_str = coordinates_elem.text.strip()
                
                # Parse coordinates into tuples of (longitude, latitude)
                coord_pairs = []
                for coord in coord_str.split():
                    if coord:
                        parts = coord.split(',')
                        if len(parts) >= 2:
                            try:
                                lng, lat = float(parts[0]), float(parts[1])
                                coord_pairs.append((lng, lat))
                            except ValueError:
                                continue  # Skip invalid coordinates
                
                if len(coord_pairs) >= 3:  # Need at least 3 points for a valid polygon
                    # Create polygon and store it
                    try:
                        polygon = Polygon(coord_pairs)
                        
                        # Validate the polygon
                        if polygon.is_valid:
                            territories[territory_name] = polygon
                            valid_polygons += 1
                        else:
                            # Try to fix the polygon
                            fixed_polygon = polygon.buffer(0)
                            if fixed_polygon.is_valid:
                                territories[territory_name] = fixed_polygon
                                valid_polygons += 1
                                st.warning(f"Fixed invalid polygon for {territory_name}")
                            else:
                                st.warning(f"Invalid polygon for {territory_name} could not be fixed")
                    except Exception as e:
                        st.warning(f"Error creating polygon for {territory_name}: {str(e)}")
                else:
                    st.warning(f"Not enough coordinate pairs for {territory_name}: {len(coord_pairs)}")
            except Exception as e:
                st.warning(f"Error processing coordinates for {territory_name}: {str(e)}")
    
    if duplicate_territories > 0:
        st.warning(f"Renamed {duplicate_territories} territories to ensure unique names")
    
    st.info(f"Successfully loaded {valid_polygons} valid polygons out of {total_polygons} total")
    return territories

def contains_with_buffer(polygon, point, buffer_distance=1e-6):
    """
    Check if a point is within a polygon with a small buffer to handle edge cases.
    This is more robust for points that are exactly on the boundary.
    Default buffer increased from 1e-10 to 1e-6 for better handling of edge cases.
    """
    try:
        # First check direct containment
        if polygon.contains(point):
            return True
        
        # If not directly contained, check with a tiny buffer for edge cases
        buffered_point = point.buffer(buffer_distance)
        return polygon.intersects(buffered_point)
    except Exception as e:
        # If there's an error, log it and return False
        st.warning(f"Error in contains_with_buffer: {str(e)}")
        return False

def normalize_territory_name(name):
    """
    Normalize territory name to handle different formats and encodings
    Enhanced to handle more cases of invisible characters and Unicode variants
    """
    if not name:
        return ""
    
    # Normalize Unicode characters (e.g., different forms of the same character)
    name = unicodedata.normalize('NFKC', name)
    
    # Convert to lowercase and remove all whitespace (not just trim)
    normalized = re.sub(r'\s+', ' ', name.lower()).strip()
    
    # Remove any non-alphanumeric characters except numbers and spaces
    normalized = re.sub(r'[^\w\d\s]', '', normalized, flags=re.UNICODE)
    
    return normalized

def find_best_territory_match(territory_name, available_territories):
    """
    Find the best match for a territory name from available territories
    Enhanced to handle more sophisticated matching
    """
    # If exact match exists, return it
    if territory_name in available_territories:
        return territory_name

    # Normalize the territory name for matching
    normalized_name = normalize_territory_name(territory_name)
    
    # Create a map of normalized names to original territory names
    normalized_map = {normalize_territory_name(t): t for t in available_territories}
    
    # Check for exact match with normalized name
    if normalized_name in normalized_map:
        return normalized_map[normalized_name]
    
    # Try fuzzy matching with a higher threshold
    best_matches = difflib.get_close_matches(
        normalized_name, 
        normalized_map.keys(),
        n=1,  # Return only the best match
        cutoff=0.7  # Increased threshold for better match quality (0.0-1.0)
    )
    
    if best_matches:
        return normalized_map[best_matches[0]]
    
    # Try a more lenient fuzzy match if no match is found
    best_matches = difflib.get_close_matches(
        normalized_name, 
        normalized_map.keys(),
        n=1,  # Return only the best match
        cutoff=0.6  # Lower threshold as a fallback
    )
    
    if best_matches:
        return normalized_map[best_matches[0]]
    
    # No good match found
    return None

def get_territory_name_mapping(kml_territories, excel_territories):
    """
    Create a mapping between KML territory names and Excel territory names
    """
    mapping = {}
    
    # First try exact matches
    for excel_territory in excel_territories:
        if excel_territory in kml_territories:
            mapping[excel_territory] = excel_territory
        else:
            # Try to find the best match
            match = find_best_territory_match(excel_territory, kml_territories)
            if match:
                mapping[excel_territory] = match
    
    return mapping

def debug_specific_polygon(territories, polygon_name, test_addresses):
    """
    Special debugging function for a specific polygon and set of addresses
    """
    if polygon_name not in territories:
        st.error(f"Polygon '{polygon_name}' not found in territories!")
        variant_names = [name for name in territories.keys() if polygon_name.lower() in name.lower()]
        if variant_names:
            st.info(f"Found similar names: {', '.join(variant_names)}")
        return
        
    polygon = territories[polygon_name]
    st.success(f"Found polygon: {polygon_name}")
    
    # Display polygon properties
    bounds = polygon.bounds
    st.write(f"Bounds: min_lng={bounds[0]:.6f}, min_lat={bounds[1]:.6f}, max_lng={bounds[2]:.6f}, max_lat={bounds[3]:.6f}")
    st.write(f"Area: {polygon.area:.8f} square degrees")
    st.write(f"Valid geometry: {polygon.is_valid}")
    
    # Test each address
    st.subheader("Testing addresses")
    for address in test_addresses:
        lat = address["lat"]
        lng = address["lng"]
        name = address["name"]
        
        point = Point(lng, lat)
        direct_match = polygon.contains(point)
        
        st.write(f"**{name}**: ({lat}, {lng})")
        st.write(f"Direct containment (no buffer): {'✅ Inside' if direct_match else '❌ Outside'}")
        
        # Test with increasing buffer sizes
        st.write("Testing with different buffer sizes:")
        for buffer in [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]:
            buffered_match = contains_with_buffer(polygon, point, buffer_distance=buffer)
            st.write(f"  Buffer {buffer}: {'✅ Inside' if buffered_match else '❌ Outside'}")
            
        # Distance to polygon
        distance = polygon.distance(point)
        st.write(f"Distance to polygon boundary: {distance*111000:.2f} meters")
        st.write("---")
    
    # Create visualization
    st.subheader("Visualization")
    # Convert polygon to GeoJSON
    polygon_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": polygon_name},
                "geometry": mapping(polygon)
            }
        ]
    })
    
    # Convert addresses to GeoJSON
    points_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": addr["name"]},
                "geometry": {
                    "type": "Point",
                    "coordinates": [addr["lng"], addr["lat"]]
                }
            } for addr in test_addresses
        ]
    })
    
    # Create map
    m_width = 700
    m_height = 500
    
    components.html(
        f"""
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <div id="debug_map" style="height: {m_height}px; width: {m_width}px;"></div>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script>
            var map = L.map('debug_map').setView([{test_addresses[0]['lat']}, {test_addresses[0]['lng']}], 14);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);
            
            // Add the polygon
            var polygon = {polygon_geojson};
            
            L.geoJSON(polygon, {{
                style: function(feature) {{
                    return {{
                        color: "#FF0000",
                        weight: 3,
                        opacity: 0.7,
                        fillOpacity: 0.3
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    if (feature.properties && feature.properties.name) {{
                        layer.bindPopup(feature.properties.name);
                    }}
                }}
            }}).addTo(map);
            
            // Add the points
            var points = {points_geojson};
            
            L.geoJSON(points, {{
                pointToLayer: function(feature, latlng) {{
                    return L.circleMarker(latlng, {{
                        radius: 8,
                        fillColor: "#0000FF",
                        color: "#000",
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }});
                }},
                onEachFeature: function(feature, layer) {{
                    if (feature.properties && feature.properties.name) {{
                        layer.bindPopup(feature.properties.name);
                    }}
                }}
            }}).addTo(map);
            
            // Fit bounds to include both polygon and points
            var bounds = L.geoJSON(polygon).getBounds();
            map.fitBounds(bounds);
        </script>
        """,
        height=m_height,
        width=m_width,
    )

def debug_territory_matching(territories, lat, lng):
    """
    Helper function to debug if a point falls within any territory
    """
    point = Point(lng, lat)
    matches = []
    
    for name, polygon in territories.items():
        try:
            if contains_with_buffer(polygon, point):
                matches.append(name)
        except Exception as e:
            st.error(f"Error checking {name}: {str(e)}")
    
    if matches:
        st.success(f"Point ({lat}, {lng}) falls within territories: {', '.join(matches)}")
    else:
        st.warning(f"Point ({lat}, {lng}) does not fall within any territory")
        
    # Debug visualization of the point
    st.write(f"Point coordinates: {point.wkt}")
    
    # For any territory that's close but not containing the point, show the distance
    st.write("Distance to territory boundaries:")
    for name, polygon in territories.items():
        try:
            distance = point.distance(polygon.boundary)
            st.write(f"• {name}: {distance:.8f} degrees (approx. {distance*111000:.2f} meters)")
        except Exception as e:
            st.write(f"• {name}: Error calculating distance - {str(e)}")
    
    return matches

def analyze_territory_names(territories):
    """
    Analyze territory names for potential issues
    """
    st.subheader("Territory Name Analysis")
    
    # Check for names that differ only in whitespace or case
    normalized_map = {}
    potential_conflicts = []
    
    for name in territories.keys():
        # Simple normalization (lowercase, remove whitespace)
        simple_norm = re.sub(r'\s+', '', name.lower())
        
        if simple_norm in normalized_map:
            existing_name = normalized_map[simple_norm]
            potential_conflicts.append((existing_name, name))
        else:
            normalized_map[simple_norm] = name
    
    if potential_conflicts:
        st.warning("Found territory names that differ only in case or whitespace:")
        for original, similar in potential_conflicts:
            st.write(f"• '{original}' vs '{similar}'")
            
            # Show exact byte representation for debugging
            st.write(f"  - '{original}': {[ord(c) for c in original]}")
            st.write(f"  - '{similar}': {[ord(c) for c in similar]}")
    else:
        st.success("No territory names with whitespace or case conflicts found")
    
    # Look for name patterns in problem territories
    problem_territories = ["Павленко-Соболєв Є.Г.", "Лихачева А 3", "Гресь 3", "Шелухін 8", "Гресь 4", "Гресь десн"]
    
    st.write("Checking for problem territory name patterns:")
    for problem in problem_territories:
        found = False
        exact_match = problem in territories
        
        if exact_match:
            st.success(f"• '{problem}' - Exact match found in KML territories")
            found = True
        else:
            # Look for similar names
            matches = []
            for name in territories.keys():
                # Check if it contains the base part of the problem name
                base_name = re.sub(r'\d+', '', problem).strip()
                if base_name in name:
                    matches.append(name)
            
            if matches:
                st.warning(f"• '{problem}' - No exact match, but found similar names: {', '.join(matches)}")
                found = True
        
        if not found:
            st.error(f"• '{problem}' - No match or similar name found in KML territories")

def fix_polygon_coordinates(territories, territory_name, coordinate_offset=(0, 0)):
    """
    Apply a coordinate offset to fix polygon coordinates
    """
    if territory_name not in territories:
        st.error(f"Territory '{territory_name}' not found")
        return territories, False
    
    try:
        polygon = territories[territory_name]
        lng_offset, lat_offset = coordinate_offset
        
        # Get coordinates, apply offset, create new polygon
        coords = list(polygon.exterior.coords)
        new_coords = [(x + lng_offset, y + lat_offset) for x, y in coords]
        
        new_polygon = Polygon(new_coords)
        if not new_polygon.is_valid:
            new_polygon = new_polygon.buffer(0)  # Try to fix invalid geometry
        
        if new_polygon.is_valid:
            territories[territory_name] = new_polygon
            st.success(f"Successfully adjusted coordinates for '{territory_name}'")
            return territories, True
        else:
            st.error(f"Failed to create valid polygon after coordinate adjustment for '{territory_name}'")
            return territories, False
    except Exception as e:
        st.error(f"Error fixing polygon coordinates: {str(e)}")
        return territories, False

def debug_problematic_addresses(territories, addresses, buffer_distance=1e-6):
    """
    Debug specific problematic addresses to see why they don't match any territory
    """
    st.subheader("Debugging Problematic Addresses")
    
    matched_addresses = []
    unmatched_addresses = []
    
    # Check each address
    for address in addresses:
        lat = address["lat"]
        lng = address["lng"]
        name = address["name"]
        
        point = Point(lng, lat)
        matched = False
        
        # Check all territories
        for territory_name, polygon in territories.items():
            try:
                if contains_with_buffer(polygon, point, buffer_distance):
                    matched_addresses.append({
                        "address": address,
                        "territory": territory_name
                    })
                    matched = True
                    break
            except Exception as e:
                st.error(f"Error checking {name} with {territory_name}: {str(e)}")
        
        # If not matched to any territory
        if not matched:
            unmatched_addresses.append(address)
    
    # Display results
    st.write(f"Checked {len(addresses)} addresses with buffer distance {buffer_distance}:")
    st.write(f"• {len(matched_addresses)} addresses matched to a territory")
    st.write(f"• {len(unmatched_addresses)} addresses did not match any territory")
    
    # Show unmatched addresses
    if unmatched_addresses:
        st.subheader("Unmatched Addresses")
        
        # Find closest territory for each unmatched address
        for address in unmatched_addresses:
            point = Point(address["lng"], address["lat"])
            closest_territory = None
            closest_distance = float('inf')
            
            for territory_name, polygon in territories.items():
                try:
                    distance = polygon.distance(point)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_territory = territory_name
                except Exception:
                    pass
            
            st.write(f"**{address['name']}** ({address['lat']}, {address['lng']})")
            if closest_territory:
                st.write(f"• Closest territory: {closest_territory}")
                st.write(f"• Distance: {closest_distance*111000:.2f} meters")
            else:
                st.write("• Could not determine closest territory")
            
            # Test with different buffer distances
            st.write("• Testing with larger buffer distances:")
            for buffer in [1e-5, 1e-4, 1e-3, 1e-2]:
                matched_territory = None
                for territory_name, polygon in territories.items():
                    try:
                        if contains_with_buffer(polygon, point, buffer):
                            matched_territory = territory_name
                            break
                    except Exception:
                        pass
                
                result = f"✅ Matches with {matched_territory}" if matched_territory else "❌ No match"
                st.write(f"  Buffer {buffer}: {result}")
            
            st.write("---")
    
    # Visualize unmatched addresses
    if unmatched_addresses:
        st.subheader("Visualization of Unmatched Addresses")
        
        # Convert all territories to GeoJSON
        all_territories_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": name},
                    "geometry": mapping(polygon)
                } for name, polygon in territories.items()
            ]
        }
        
        # Convert unmatched addresses to GeoJSON
        points_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": addr["name"]},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [addr["lng"], addr["lat"]]
                    }
                } for addr in unmatched_addresses
            ]
        }
        
        # Create map centered on unmatched addresses
        center_lat = sum(addr["lat"] for addr in unmatched_addresses) / len(unmatched_addresses)
        center_lng = sum(addr["lng"] for addr in unmatched_addresses) / len(unmatched_addresses)
        
        m_width = 700
        m_height = 500
        
        components.html(
            f"""
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <div id="unmatched_map" style="height: {m_height}px; width: {m_width}px;"></div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                var map = L.map('unmatched_map').setView([{center_lat}, {center_lng}], 12);
                
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }}).addTo(map);
                
                // Add all territories
                var territories = {json.dumps(all_territories_geojson)};
                
                var territoryLayer = L.geoJSON(territories, {{
                    style: function(feature) {{
                        return {{
                            color: "#FF0000",
                            weight: 2,
                            opacity: 0.5,
                            fillOpacity: 0.2
                        }};
                    }},
                    onEachFeature: function(feature, layer) {{
                        if (feature.properties && feature.properties.name) {{
                            layer.bindPopup(feature.properties.name);
                        }}
                    }}
                }}).addTo(map);
                
                // Add unmatched points
                var points = {json.dumps(points_geojson)};
                
                var pointsLayer = L.geoJSON(points, {{
                    pointToLayer: function(feature, latlng) {{
                        return L.circleMarker(latlng, {{
                            radius: 8,
                            fillColor: "#0000FF",
                            color: "#000",
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        }});
                    }},
                    onEachFeature: function(feature, layer) {{
                        if (feature.properties && feature.properties.name) {{
                            layer.bindPopup(feature.properties.name);
                        }}
                    }}
                }}).addTo(map);
                
                // Fit map to include points
                var bounds = pointsLayer.getBounds();
                map.fitBounds(bounds);
            </script>
            """,
            height=m_height,
            width=m_width,
        )
    
    return matched_addresses, unmatched_addresses

def main():
    st.set_page_config(page_title="Territory Analyzer", page_icon="🗺️", layout="wide")
    
    st.title("Territory Analyzer with Specific Debug for Перова Addresses")
    st.markdown("---")
    
    # Embed the Google Map
    map_id = "17-ck1hNUEqZ02FpNcW6fiva9UGQhypM"
    components.iframe(
        f"https://www.google.com/maps/d/embed?mid={map_id}",
        height=600,
        scrolling=True
    )
    
    st.markdown("---")
    st.header("Process Data with Territory Analysis")
    
    # Step 1: Upload KML/KMZ file with territories
    st.subheader("Step 1: Upload KML/KMZ File with Territory Definitions")
    kml_file = st.file_uploader("Upload KML or KMZ file exported from Google Maps", type=["kml", "kmz"])
    
    territories = {}
    if kml_file is not None:
        try:
            # Check if file is KMZ (zip) or KML
            file_content = kml_file.read()
            kml_content = ""
            
            # If KMZ (zip file)
            if kml_file.name.lower().endswith('.kmz'):
                try:
                    # Extract the KML from the KMZ (which is a zip file)
                    with zipfile.ZipFile(BytesIO(file_content)) as kmz:
                        kml_files = [f for f in kmz.namelist() if f.lower().endswith('.kml')]
                        if kml_files:
                            # Usually the main KML file is doc.kml
                            main_kml = 'doc.kml' if 'doc.kml' in kml_files else kml_files[0]
                            kml_content = kmz.read(main_kml).decode('utf-8')
                        else:
                            st.error("No KML file found in the KMZ archive")
                except Exception as e:
                    st.error(f"Error extracting KML from KMZ: {str(e)}")
            else:
                # Regular KML file
                kml_content = file_content.decode('utf-8')
            
            if kml_content:
                territories = extract_coordinates_from_kml(kml_content)
                
                # Display the territory names
                if territories:
                    st.success(f"Successfully loaded {len(territories)} territories from {kml_file.name}!")
                    
                    # Analyze territory names for potential issues
                    analyze_territory_names(territories)

                    # Special debug section for Перова addresses and Гресь десн polygon
                    with st.expander("Debug Specific Case: Перова addresses and Гресь десн polygon"):
                        st.write("This section specifically debugs why addresses on Перова street aren't matching with the Гресь десн polygon")
                        
                        perovа_addresses = [
                            {"name": "м. Київ, бульв. Перова, 15", "lat": 50.48234, "lng": 30.59336},
                            {"name": "м. Київ, бульв. Перова, 20", "lat": 50.47779, "lng": 30.59606}
                        ]
                        
                        # Try to find "Гресь десн" with exact match or similar
                        target_polygon = "Гресь десн"
                        if target_polygon not in territories:
                            # Try to find a similar name
                            similar_names = [name for name in territories.keys() if "гресь" in name.lower() and "десн" in name.lower()]
                            if similar_names:
                                st.info(f"Exact name 'Гресь десн' not found, but found similar names: {', '.join(similar_names)}")
                                target_polygon = similar_names[0]  # Use the first similar name
                            else:
                                # Try just "десн"
                                similar_names = [name for name in territories.keys() if "десн" in name.lower()]
                                if similar_names:
                                    st.info(f"Found names with 'десн': {', '.join(similar_names)}")
                                    target_polygon = similar_names[0]
                                else:
                                    # Try just "гресь"
                                    similar_names = [name for name in territories.keys() if "гресь" in name.lower()]
                                    if similar_names:
                                        st.info(f"Found names with 'гресь': {', '.join(similar_names)}")
                                        target_polygon = similar_names[0]
                        
                        if target_polygon in territories:
                            debug_specific_polygon(territories, target_polygon, perovа_addresses)
                        else:
                            st.error("Could not find 'Гресь десн' or any similar name in the territories")
                            
                            # Show all territory names for reference
                            with st.expander("Show all territory names"):
                                for name in sorted(territories.keys()):
                                    st.write(f"• {name}")
                    
                    # Debug problematic addresses
                    with st.expander("Debug All Problematic Addresses"):
                        st.write("This section tests all problematic addresses against the territories")
                        
                        # List of problematic addresses
                        problem_addresses = [
                            {"name": "м. Київ, бульв. Перова, 15", "lat": 50.48234, "lng": 30.59336},
                            {"name": "м. Київ, бульв. Перова, 20", "lat": 50.47779, "lng": 30.59606},
                            {"name": "м. Київ , вул. Юності, 6а (літера А)", "lat": 50.46397, "lng": 30.61831},
                            {"name": "м. Київ , шосе Харківське, 17 а, прим.101", "lat": 50.43375, "lng": 30.63452},
                            {"name": "м. Київ, б-р Бучми Амвросія, 6а", "lat": 50.43183, "lng": 30.60336},
                            {"name": "м. Київ, вул. Алма-Атинська, 39-3", "lat": 50.44334, "lng": 30.66523},
                            {"name": "м. Київ, вул. Березнева, 12 А", "lat": 50.43195, "lng": 30.62288},
                            {"name": "м. Київ, вул. Березняківська, 30-Б", "lat": 50.42758, "lng": 30.61138},
                            {"name": "м. Київ, вул. Запорожця Петра, 26", "lat": 50.47898, "lng": 30.60603},
                            {"name": "м. Київ, вул. Каховська, 56", "lat": 50.46196, "lng": 30.59185},
                            {"name": "м. Київ, вул. Кибальчича Миколи, 11В", "lat": 50.49304, "lng": 30.59856},
                            {"name": "м. Київ, вул. Миропільська, 1", "lat": 50.46269, "lng": 30.62768},
                            {"name": "м. Київ, вул. Празька, 24", "lat": 50.43779, "lng": 30.64064},
                            {"name": "м. Київ, набережна Русанівська, будинок 4", "lat": 50.44092, "lng": 30.59334},
                            {"name": "м. Київ, просп. Соборності, 5", "lat": 50.44021, "lng": 30.62023}
                        ]
                        
                        # Test with different buffer sizes
                        buffer_options = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
                        selected_buffer = st.selectbox(
                            "Select buffer distance for testing",
                            options=buffer_options,
                            index=buffer_options.index(1e-6),  # Default 1e-6
                            format_func=lambda x: f"{x} (approx. {x*111000:.2f} meters)"
                        )
                        
                        # Run the debug
                        matched, unmatched = debug_problematic_addresses(territories, problem_addresses, selected_buffer)
                        
                        # Show coordinate adjustment options if there are unmatched addresses
                        if unmatched:
                            st.subheader("Coordinate Adjustment Options")
                            st.write("If addresses aren't matching due to coordinate system differences, you can try applying an offset")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                lng_offset = st.number_input("Longitude offset", value=0.0, step=0.0001, format="%.6f")
                            with col2:
                                lat_offset = st.number_input("Latitude offset", value=0.0, step=0.0001, format="%.6f")
                            
                            if st.button("Apply offset and test again"):
                                # Create a copy of territories with adjusted coordinates
                                adjusted_territories = {}
                                for name, polygon in territories.items():
                                    # Extract coordinates and apply offset
                                    coords = list(polygon.exterior.coords)
                                    new_coords = [(x + lng_offset, y + lat_offset) for x, y in coords]
                                    
                                    try:
                                        new_polygon = Polygon(new_coords)
                                        if not new_polygon.is_valid:
                                            new_polygon = new_polygon.buffer(0)  # Try to fix invalid geometry
                                        
                                        if new_polygon.is_valid:
                                            adjusted_territories[name] = new_polygon
                                    except Exception as e:
                                        st.error(f"Error adjusting polygon {name}: {str(e)}")
                                
                                if adjusted_territories:
                                    st.success(f"Applied offset ({lng_offset}, {lat_offset}) to {len(adjusted_territories)} territories")
                                    matched_adj, unmatched_adj = debug_problematic_addresses(
                                        adjusted_territories, problem_addresses, selected_buffer
                                    )
                                    
                                    # Compare results
                                    if len(unmatched_adj) < len(unmatched):
                                        st.success(f"✅ Improvement! Unmatched addresses reduced from {len(unmatched)} to {len(unmatched_adj)}")
                                    elif len(unmatched_adj) > len(unmatched):
                                        st.error(f"❌ Worse! Unmatched addresses increased from {len(unmatched)} to {len(unmatched_adj)}")
                                    else:
                                        st.warning(f"⚠️ No change in number of unmatched addresses ({len(unmatched)})")
                else:
                    st.warning("No valid territories were extracted from the KML file")
        except Exception as e:
            st.error(f"Error parsing KML/KMZ file: {str(e)}")
            st.exception(e)
    
    # Step 2: Upload Excel file with coordinates
    st.subheader("Step 2: Upload Excel File with Coordinates")
    excel_file = st.file_uploader("Upload CONSO.xlsx file", type=["xlsx"])
    
    if excel_file is not None:
        try:
            df = pd.read_excel(excel_file)
            st.success("Excel file successfully uploaded!")
            
            # Display preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Detect or select latitude and longitude columns
            if 'latitude' in df.columns and 'longitude' in df.columns:
                lat_column = 'latitude'
                lng_column = 'longitude'
                st.info(f"Found latitude and longitude columns automatically.")
            else:
                # Let user select columns
                st.warning("Could not find standard latitude and longitude columns.")
                columns = df.columns.tolist()
                lat_column = st.selectbox("Select Latitude Column", columns)
                lng_column = st.selectbox("Select Longitude Column", columns)
            
            # Check for existing territory column and collect territory names
            existing_territories = set()
            if 'territory' in df.columns:
                existing_territories = set(df['territory'].dropna().unique())
                st.info(f"Found {len(existing_territories)} existing territories in Excel file.")
                
                with st.expander("View existing territories in Excel"):
                    for territory in sorted(existing_territories):
                        count = df[df['territory'] == territory].shape[0]
                        st.write(f"• **{territory}**: {count} addresses")
                
                # Create territory name mapping
                if territories and existing_territories:
                    territory_mapping = get_territory_name_mapping(territories.keys(), existing_territories)
                    
                    if territory_mapping:
                        st.success(f"Created mapping for {len(territory_mapping)} territories between Excel and KML")
                        
                        with st.expander("View territory name mapping"):
                            for excel_name, kml_name in territory_mapping.items():
                                if excel_name != kml_name:
                                    st.write(f"• Excel: **{excel_name}** → KML: **{kml_name}**")
            
            # Add options for territory matching
            st.subheader("Territory Matching Options")
            
            col1, col2 = st.columns(2)
            with col1:
                prioritize_specific = st.checkbox("Prioritize specific territories when overlapping", value=True)
                if prioritize_specific:
                    # Get all territory names for selection
                    all_territories = list(territories.keys())
                    if "Шелухін" in all_territories:
                        default_priority = "Шелухін"
                    elif "Шелухін 7" in all_territories:
                        default_priority = "Шелухін 7"
                    elif len(all_territories) > 0:
                        default_priority = all_territories[0]
                    else:
                        default_priority = ""
                    
                    priority_territory = st.selectbox(
                        "Select territory to prioritize",
                        options=all_territories,
                        index=all_territories.index(default_priority) if default_priority in all_territories and len(all_territories) > 0 else 0
                    )
            
            with col2:
                handle_boundaries = st.checkbox("Handle boundary cases with tolerance", value=True)
                if handle_boundaries:
                    boundary_tolerance = st.number_input(
                        "Boundary tolerance (smaller = more precise)", 
                        min_value=1e-10, 
                        max_value=1e-4, 
                        value=1e-5,  # Increased default from 1e-6 to 1e-5
                        format="%.10f"
                    )
                    
                    st.info(f"A buffer of {boundary_tolerance} is approximately {boundary_tolerance*111000:.2f} meters")
            
            # Add name matching options
            with st.expander("Advanced Territory Name Matching"):
                use_name_mapping = st.checkbox("Use territory name mapping", value=True)
                st.info("This option will map territory names in the Excel file to the closest matching territory names in the KML file")
                
                fuzzy_matching = st.checkbox("Use fuzzy name matching for similar territory names", value=True)
                st.info("This option will try to match similar territory names even if they have slight differences in spelling")
                
                # Option to preserve existing territories
                preserve_existing = st.checkbox("Preserve existing territory assignments", value=True)
                st.info("This option will keep existing territory assignments in the Excel file and only assign territories to addresses that don't have one")
            
            # Process button (only active if territories are loaded)
            process_button = st.button("Analyze Territories", disabled=len(territories) == 0)
            
            if len(territories) == 0 and kml_file is not None:
                st.warning("No territories were extracted from the KML file. Please check the file format.")
            
            if process_button:
                with st.spinner("Processing data..."):
                    # Ensure lat/long are numeric
                    df[lat_column] = pd.to_numeric(df[lat_column], errors='coerce')
                    df[lng_column] = pd.to_numeric(df[lng_column], errors='coerce')
                    
                    # Create a copy of the original territory column if it exists
                    if 'territory' in df.columns and preserve_existing:
                        df['original_territory'] = df['territory'].copy()
                    
                    # Build territory name mapping if needed
                    territory_mapping = {}
                    if 'territory' in df.columns and use_name_mapping:
                        excel_territories = set(df['territory'].dropna().unique())
                        kml_territories = set(territories.keys())
                        
                        for excel_name in excel_territories:
                            if excel_name in kml_territories:
                                territory_mapping[excel_name] = excel_name
                            else:
                                if fuzzy_matching:
                                    match = find_best_territory_match(excel_name, kml_territories)
                                    if match:
                                        territory_mapping[excel_name] = match
                    
                    # Determine territory for each point
                    territory_results = []
                    
                    # Create a progress bar for visual feedback
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    
                    # For debugging purpose, count addresses in territories
                    territory_match_count = {name: 0 for name in territories.keys()}
                    territory_match_count["Outside territory"] = 0
                    territory_match_count["Preserved original"] = 0
                    
                    # Store details about which addresses didn't match any territory
                    unmatched_addresses = []
                    
                    # Store details about multi-match cases
                    multi_match_cases = []
                    
                    for idx, row in df.iterrows():
                        # Update progress bar
                        if idx % 100 == 0:
                            progress_bar.progress(min(idx / total_rows, 1.0))
                            
                        # Check if we should preserve the existing territory
                        if 'original_territory' in df.columns and preserve_existing and pd.notna(row['original_territory']):
                            original_territory = row['original_territory']
                            
                            # If we have a mapping for this territory name, use the mapped KML territory name
                            if use_name_mapping and original_territory in territory_mapping:
                                mapped_territory = territory_mapping[original_territory]
                                territory_results.append(mapped_territory)
                                territory_match_count["Preserved original"] += 1
                            else:
                                # Use the original territory name
                                territory_results.append(original_territory)
                                territory_match_count["Preserved original"] += 1
                            
                            continue
                        
                        lat = row[lat_column]
                        lng = row[lng_column]
                        
                        if pd.notnull(lat) and pd.notnull(lng):
                            point = Point(lng, lat)
                            matched_territories = []
                            
                            # Check all territories without breaking on first match
                            for name, polygon in territories.items():
                                try:
                                    # Use the enhanced containment check with buffer for edge cases
                                    if handle_boundaries:
                                        if contains_with_buffer(polygon, point, boundary_tolerance):
                                            matched_territories.append(name)
                                    else:
                                        if polygon.contains(point):
                                            matched_territories.append(name)
                                except Exception as e:
                                    st.warning(f"Error checking if point is in territory {name}: {str(e)}")
                            
                            # Record multi-match cases for analysis
                            if len(matched_territories) > 1:
                                address_info = {
                                    'address': row.get('Address new', row.get('address', "Unknown")),
                                    'lat': lat,
                                    'lng': lng,
                                    'territories': matched_territories
                                }
                                multi_match_cases.append(address_info)
                            
                            # If we found any territories, determine which one to use
                            if matched_territories:
                                if prioritize_specific and priority_territory in matched_territories:
                                    # Prioritize the specified territory
                                    matched_territory = priority_territory
                                else:
                                    # Use the first matching territory
                                    matched_territory = matched_territories[0]
                                
                                territory_match_count[matched_territory] += 1
                                territory_results.append(matched_territory)
                            else:
                                matched_territory = None
                                territory_match_count["Outside territory"] += 1
                                territory_results.append(matched_territory)
                                
                                # Store information about unmatched address for debugging
                                if len(unmatched_addresses) < 100:  # Limit to 100 examples
                                    address_info = {
                                        'address': row.get('Address new', row.get('address', "Unknown")),
                                        'lat': lat,
                                        'lng': lng
                                    }
                                    unmatched_addresses.append(address_info)
                        else:
                            matched_territory = None
                            territory_match_count["Outside territory"] += 1
                            territory_results.append(matched_territory)
                    
                    # Complete the progress bar
                    progress_bar.progress(1.0)
                    
                    # Add territory column to dataframe
                    df['territory'] = territory_results
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.dataframe(df)
                    
                    # Create download button for the processed file
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="Download Processed Excel File",
                        data=output,
                        file_name="territory_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Display statistics
                    st.subheader("Territory Statistics")
                    territory_counts = df['territory'].value_counts(dropna=False).reset_index()
                    territory_counts.columns = ['Territory', 'Count']
                    
                    # Replace None with "Outside territory"
                    territory_counts['Territory'] = territory_counts['Territory'].fillna("Outside territory")
                    
                    if not territory_counts.empty:
                        # Create a bar chart
                        st.bar_chart(territory_counts.set_index('Territory'))
                        
                        # Display territory distribution as text
                        st.subheader("Summary:")
                        for i, row in territory_counts.iterrows():
                            st.write(f"• {row['Territory']}: {row['Count']} locations")
                        
                        total = territory_counts['Count'].sum()
                        st.write(f"**Total: {total} locations**")
                    
                    # Debug information about potentially problematic polygons
                    st.subheader("Debug Information")
                    
                    # Show overlapping territory information
                    if multi_match_cases:
                        with st.expander(f"Overlapping Territories ({len(multi_match_cases)} addresses)"):
                            st.write("These addresses fall within multiple territories:")
                            for i, case in enumerate(multi_match_cases[:10]):  # Show max 10 examples
                                st.write(f"• **{case['address']}** ({case['lat']}, {case['lng']})")
                                st.write(f"  Territories: {', '.join(case['territories'])}")
                            
                            if len(multi_match_cases) > 10:
                                st.write(f"... and {len(multi_match_cases) - 10} more addresses with multiple territories")
                    
                    # Show territories with no matches
                    empty_territories = [name for name, count in territory_match_count.items() 
                                        if count == 0 and name != "Outside territory" and name != "Preserved original"]
                    if empty_territories:
                        st.warning(f"Territories with no matching addresses: {', '.join(empty_territories)}")
                
                    # Show information about unmatched addresses
                    if unmatched_addresses:
                        with st.expander(f"Unmatched Addresses (Sample of {len(unmatched_addresses)})"):
                            st.write(f"Total unmatched addresses: {territory_match_count['Outside territory']}")
                            
                            # Create a button to debug these unmatched addresses
                            if st.button("Debug Unmatched Addresses"):
                                problem_addresses = [
                                    {
                                        "name": addr["address"],
                                        "lat": addr["lat"],
                                        "lng": addr["lng"]
                                    } for addr in unmatched_addresses
                                ]
                                
                                # Run detailed debug on these addresses
                                debug_problematic_addresses(territories, problem_addresses, boundary_tolerance)
        
        except Exception as e:
            st.error(f"Error processing the Excel file: {str(e)}")
            st.exception(e)
    
    # Instructions for exporting KML
    if kml_file is None:
        st.markdown("---")
        st.markdown("""
        ### How to export your Google Maps territories as KML/KMZ:
        
        1. Open your Google Map in My Maps: [https://www.google.com/maps/d/edit?mid=17-ck1hNUEqZ02FpNcW6fiva9UGQhypM](https://www.google.com/maps/d/edit?mid=17-ck1hNUEqZ02FpNcW6fiva9UGQhypM)
        2. Click the three dots (⋮) next to your map name
        3. Select "Export to KML/KMZ" as shown in your screenshot
        4. Choose to export all features or specific layers
        5. Save the KML or KMZ file to your computer
        6. Upload the file here
        """)

if __name__ == "__main__":
    main()
