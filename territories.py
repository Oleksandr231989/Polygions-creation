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

def extract_coordinates_from_kml(kml_content):
    """
    Extract polygon coordinates from KML content with improved error handling and debugging
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
    
    # Find all Placemark elements
    placemarks = root.findall('.//kml:Placemark', ns)
    
    total_polygons = 0
    valid_polygons = 0
    st.info(f"Found {len(placemarks)} placemarks in KML file")
    
    for placemark in placemarks:
        # Get the name of the territory
        name_elem = placemark.find('kml:name', ns)
        if name_elem is not None:
            territory_name = name_elem.text.strip()
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
    
    st.info(f"Successfully loaded {valid_polygons} valid polygons out of {total_polygons} total")
    return territories

def contains_with_buffer(polygon, point, buffer_distance=1e-10):
    """
    Check if a point is within a polygon with a small buffer to handle edge cases.
    This is more robust for points that are exactly on the boundary.
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
    """
    if not name:
        return ""
    
    # Convert to lowercase and remove spaces
    normalized = name.lower().strip()
    
    # Remove any non-alphanumeric characters except numbers
    normalized = re.sub(r'[^\w\d\s]', '', normalized, flags=re.UNICODE)
    
    return normalized

def find_best_territory_match(territory_name, available_territories):
    """
    Find the best match for a territory name from available territories
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
    
    # Try fuzzy matching
    best_matches = difflib.get_close_matches(
        normalized_name, 
        normalized_map.keys(),
        n=1,  # Return only the best match
        cutoff=0.6  # Adjust this threshold for match quality (0.0-1.0)
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
            st.write(f"• {name}: {distance:.8f} degrees")
        except Exception as e:
            st.write(f"• {name}: Error calculating distance - {str(e)}")
    
    return matches

def export_territories_geojson(territories):
    """
    Export territories as GeoJSON for visualization
    """
    features = []
    for name, polygon in territories.items():
        feature = {
            "type": "Feature",
            "properties": {"name": name},
            "geometry": mapping(polygon)
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return json.dumps(geojson)

def preview_address_points(df, lat_column, lng_column, max_points=100):
    """
    Generate a GeoJSON of address points for preview
    """
    features = []
    df_sample = df.dropna(subset=[lat_column, lng_column]).sample(min(max_points, len(df)))
    
    for _, row in df_sample.iterrows():
        lat = row[lat_column]
        lng = row[lng_column]
        if pd.notnull(lat) and pd.notnull(lng):
            feature = {
                "type": "Feature",
                "properties": {
                    "address": row.get("Address new", "Unknown"),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]
                }
            }
            features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return json.dumps(geojson)

def main():
    st.set_page_config(page_title="Territory Analyzer", page_icon="🗺️", layout="wide")
    
    st.title("Territory Analyzer")
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
                    
                    # Create expandable sections for territories
                    with st.expander("View loaded territories"):
                        for name in sorted(territories.keys()):
                            polygon = territories[name]
                            bounds = polygon.bounds
                            st.write(f"• **{name}**: Bounds (min_lng={bounds[0]:.6f}, min_lat={bounds[1]:.6f}, max_lng={bounds[2]:.6f}, max_lat={bounds[3]:.6f})")
                    
                    # Add territory visualization option
                    if st.checkbox("Visualize territories (may be slow for many territories)"):
                        geojson_str = export_territories_geojson(territories)
                        m_width = 700
                        m_height = 500
                        
                        # Use Leaflet to visualize
                        components.html(
                            f"""
                            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                            <div id="map" style="height: {m_height}px; width: {m_width}px;"></div>
                            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                            <script>
                                var map = L.map('map').setView([50.45, 30.52], 10); // Centered on Kyiv
                                
                                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                }}).addTo(map);
                                
                                var geojson = {geojson_str};
                                
                                function getRandomColor() {{
                                    var letters = '0123456789ABCDEF';
                                    var color = '#';
                                    for (var i = 0; i < 6; i++) {{
                                        color += letters[Math.floor(Math.random() * 16)];
                                    }}
                                    return color;
                                }}
                                
                                L.geoJSON(geojson, {{
                                    style: function(feature) {{
                                        return {{
                                            color: getRandomColor(),
                                            weight: 2,
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
                                
                                // Fit map to data bounds
                                var bounds = L.geoJSON(geojson).getBounds();
                                if (!bounds.isValid()) {{
                                    // Default to Kyiv if bounds are invalid
                                    map.setView([50.45, 30.52], 10);
                                }} else {{
                                    map.fitBounds(bounds);
                                }}
                            </script>
                            """,
                            height=m_height,
                            width=m_width,
                        )
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
            
            # Add preview option for address points
            if st.checkbox("Preview address points on map"):
                if df[lat_column].notna().sum() > 0 and df[lng_column].notna().sum() > 0:
                    geojson_str = preview_address_points(df, lat_column, lng_column)
                    m_width = 700
                    m_height = 500
                    
                    components.html(
                        f"""
                        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                        <div id="address_map" style="height: {m_height}px; width: {m_width}px;"></div>
                        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                        <script>
                            var map = L.map('address_map').setView([50.45, 30.52], 10); // Centered on Kyiv
                            
                            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            }}).addTo(map);
                            
                            var geojson = {geojson_str};
                            
                            L.geoJSON(geojson, {{
                                pointToLayer: function(feature, latlng) {{
                                    return L.circleMarker(latlng, {{
                                        radius: 5,
                                        fillColor: "#ff7800",
                                        color: "#000",
                                        weight: 1,
                                        opacity: 1,
                                        fillOpacity: 0.8
                                    }});
                                }},
                                onEachFeature: function(feature, layer) {{
                                    if (feature.properties && feature.properties.address) {{
                                        layer.bindPopup(feature.properties.address);
                                    }}
                                }}
                            }}).addTo(map);
                            
                            // Fit map to data bounds
                            var bounds = L.geoJSON(geojson).getBounds();
                            if (!bounds.isValid()) {{
                                // Default to Kyiv if bounds are invalid
                                map.setView([50.45, 30.52], 10);
                            }} else {{
                                map.fitBounds(bounds);
                            }}
                        </script>
                        """,
                        height=m_height,
                        width=m_width,
                    )
                else:
                    st.warning("No valid coordinates found in the data")
            
            # Add debug section for testing specific coordinates
            with st.expander("Debug Territory Matching"):
                st.write("Use this tool to check if specific coordinates fall within any territory")
                test_lat = st.number_input("Test Latitude", value=50.5126704)
                test_lng = st.number_input("Test Longitude", value=30.4267758)
                debug_button = st.button("Test These Coordinates")

                if debug_button and territories:
                    debug_territory_matching(territories, test_lat, test_lng)
            
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
                        index=all_territories.index(default_priority) if default_priority in all_territories else 0
                    )
            
            with col2:
                handle_boundaries = st.checkbox("Handle boundary cases with tolerance", value=True)
                if handle_boundaries:
                    boundary_tolerance = st.number_input(
                        "Boundary tolerance (smaller = more precise)", 
                        min_value=1e-10, 
                        max_value=1e-5, 
                        value=1e-8,
                        format="%.10f"
                    )
            
            # Add name matching options
            with st.expander("Advanced Territory Name Matching"):
                use_name_mapping = st.checkbox("Use territory name mapping", value=True)
                st.info("This option will map territory names in the Excel file to the closest matching territory names in the KML file")
                
                if 'territory' in df.columns and use_name_mapping:
                    # Build and show the territory name mapping
                    excel_territories = set(df['territory'].dropna().unique())
                    kml_territories = set(territories.keys())
                    
                    # Create mapping
                    territory_mapping = {}
                    for excel_name in excel_territories:
                        if excel_name in kml_territories:
                            territory_mapping[excel_name] = excel_name
                        else:
                            match = find_best_territory_match(excel_name, kml_territories)
                            if match:
                                territory_mapping[excel_name] = match
                    
                    st.write("Territory name mapping:")
                    for excel_name, kml_name in territory_mapping.items():
                        if excel_name != kml_name:
                            st.write(f"• Excel: **{excel_name}** → KML: **{kml_name}**")
                
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
                                multi_match_cases.append({
                                    'address': row.get('Address new', 'Unknown'),
                                    'lat': lat,
                                    'lng': lng,
                                    'territories': matched_territories
                                })
                            
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
                                    unmatched_addresses.append({
                                        'address': row.get('Address new', 'Unknown'),
                                        'lat': lat,
                                        'lng': lng
                                    })
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
                            st.write("Sample of addresses that didn't match any territory:")
                            for addr in unmatched_addresses[:10]:  # Show just the first 10
                                st.write(f"• {addr['address']} ({addr['lat']}, {addr['lng']})")
                            
                            # Create a map with some unmatched addresses for visualization
                            if len(unmatched_addresses) > 0:
                                st.write("Map of some unmatched addresses:")
                                
                                # Create GeoJSON for unmatched addresses
                                features = []
                                for addr in unmatched_addresses[:50]:  # Limit to 50 for performance
                                    feature = {
                                        "type": "Feature",
                                        "properties": {"address": addr['address']},
                                        "geometry": {
                                            "type": "Point",
                                            "coordinates": [addr['lng'], addr['lat']]
                                        }
                                    }
                                    features.append(feature)
                                
                                geojson_str = json.dumps({"type": "FeatureCollection", "features": features})
                                
                                # Create Leaflet map
                                m_width = 700
                                m_height = 400
                                
                                components.html(
                                    f"""
                                    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                                    <div id="unmatched_map" style="height: {m_height}px; width: {m_width}px;"></div>
                                    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                                    <script>
                                        var map = L.map('unmatched_map').setView([50.45, 30.52], 10);
                                        
                                        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                        }}).addTo(map);
                                        
                                        var geojson = {geojson_str};
                                        
                                        L.geoJSON(geojson, {{
                                            pointToLayer: function(feature, latlng) {{
                                                return L.circleMarker(latlng, {{
                                                    radius: 5,
                                                    fillColor: "#ff0000",
                                                    color: "#000",
                                                    weight: 1,
                                                    opacity: 1,
                                                    fillOpacity: 0.8
                                                }});
                                            }},
                                            onEachFeature: function(feature, layer) {{
                                                if (feature.properties && feature.properties.address) {{
                                                    layer.bindPopup(feature.properties.address);
                                                }}
                                            }}
                                        }}).addTo(map);
                                        
                                        // Fit map to data bounds
                                        var bounds = L.geoJSON(geojson).getBounds();
                                        if (bounds.isValid()) {{
                                            map.fitBounds(bounds);
                                        }}
                                    </script>
                                    """,
                                    height=m_height,
                                    width=m_width,
                                )
                            
                            # Suggest checking these coordinates against the territories
                            st.write("\nTo check if a specific point is within any territory, use the Debug Territory Matching tool")
        
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
