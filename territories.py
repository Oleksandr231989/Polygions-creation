import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io
import xml.etree.ElementTree as ET
import re
from shapely.geometry import Point, Polygon
import base64

def extract_coordinates_from_kml(kml_content):
    """
    Extract polygon coordinates from KML content
    """
    # Parse the KML content
    root = ET.fromstring(kml_content)
    
    # Define the XML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Dictionary to store territory polygons
    territories = {}
    
    # Find all Placemark elements
    placemarks = root.findall('.//kml:Placemark', ns)
    
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
            # Extract and clean coordinate string
            coord_str = coordinates_elem.text.strip()
            
            # Parse coordinates into tuples of (longitude, latitude)
            coord_pairs = []
            for coord in coord_str.split():
                if coord:
                    parts = coord.split(',')
                    if len(parts) >= 2:
                        lng, lat = float(parts[0]), float(parts[1])
                        coord_pairs.append((lng, lat))
            
            if coord_pairs:
                # Create polygon and store it
                polygon = Polygon(coord_pairs)
                territories[territory_name] = polygon
    
    return territories

def debug_territory_matching(territories, lat, lng):
    """
    Helper function to debug if a point falls within any territory
    """
    point = Point(lng, lat)
    matches = []
    
    for name, polygon in territories.items():
        try:
            if polygon.contains(point):
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
            st.write(f"• {name}: {distance:.6f} degrees")
        except Exception as e:
            st.write(f"• {name}: Error calculating distance - {str(e)}")
    
    return matches

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
                import zipfile
                from io import BytesIO
                
                # Extract the KML from the KMZ (which is a zip file)
                with zipfile.ZipFile(BytesIO(file_content)) as kmz:
                    # Usually the main KML file is doc.kml
                    kml_content = kmz.read('doc.kml').decode('utf-8')
            else:
                # Regular KML file
                kml_content = file_content.decode('utf-8')
                
            territories = extract_coordinates_from_kml(kml_content)
            st.success(f"Successfully loaded {len(territories)} territories from {kml_file.name}!")
            
            # Display the territory names
            st.write("Territories found:")
            for name in territories.keys():
                st.write(f"• {name}")
                
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
            
            # Add debug section for testing specific coordinates
            with st.expander("Debug Territory Matching"):
                st.write("Use this tool to check if specific coordinates fall within any territory")
                test_lat = st.number_input("Test Latitude", value=50.5126704)
                test_lng = st.number_input("Test Longitude", value=30.4267758)
                debug_button = st.button("Test These Coordinates")

                if debug_button and territories:
                    debug_territory_matching(territories, test_lat, test_lng)
            
            # Process button (only active if territories are loaded)
            process_button = st.button("Analyze Territories", disabled=len(territories) == 0)
            
            if len(territories) == 0 and kml_file is not None:
                st.warning("No territories were extracted from the KML file. Please check the file format.")
            
            if process_button:
                with st.spinner("Processing data..."):
                    # Ensure lat/long are numeric
                    df[lat_column] = pd.to_numeric(df[lat_column], errors='coerce')
                    df[lng_column] = pd.to_numeric(df[lng_column], errors='coerce')
                    
                    # Determine territory for each point
                    territory_results = []
                    
                    # Create a progress bar for visual feedback
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    
                    # For debugging purpose, count addresses in territories
                    territory_match_count = {name: 0 for name in territories.keys()}
                    territory_match_count["Outside territory"] = 0
                    
                    # Store details about which addresses didn't match any territory
                    unmatched_addresses = []
                    
                    for idx, row in df.iterrows():
                        # Update progress bar
                        if idx % 100 == 0:
                            progress_bar.progress(min(idx / total_rows, 1.0))
                            
                        lat = row[lat_column]
                        lng = row[lng_column]
                        
                        if pd.notnull(lat) and pd.notnull(lng):
                            point = Point(lng, lat)
                            matched_territories = []
                            
                            # Check all territories without breaking on first match
                            for name, polygon in territories.items():
                                try:
                                    if polygon.contains(point):
                                        matched_territories.append(name)
                                except Exception as e:
                                    st.warning(f"Error checking if point is in territory {name}: {str(e)}")
                            
                            # If we found any territories, use the first one
                            # This could be modified to use a priority system if needed
                            if matched_territories:
                                # Prioritize Шелухін 7 if it's in the matches
                                if "Шелухін 7" in matched_territories:
                                    matched_territory = "Шелухін 7"
                                else:
                                    matched_territory = matched_territories[0]
                                territory_match_count[matched_territory] += 1
                            else:
                                matched_territory = None
                                territory_match_count["Outside territory"] += 1
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
                    st.write("This information can help identify issues with territory matching:")
                    
                    # Show territories with no matches
                    empty_territories = [name for name, count in territory_match_count.items() 
                                        if count == 0 and name != "Outside territory"]
                    if empty_territories:
                        st.warning(f"Territories with no matching addresses: {', '.join(empty_territories)}")
                    
                    # Show information about unmatched addresses
                    if unmatched_addresses:
                        expander = st.expander("Sample of unmatched addresses")
                        with expander:
                            st.write(f"Total unmatched addresses: {territory_match_count['Outside territory']}")
                            st.write("Sample of addresses that didn't match any territory:")
                            for addr in unmatched_addresses[:10]:  # Show just the first 10
                                st.write(f"• {addr['address']} ({addr['lat']}, {addr['lng']})")
                            
                            # Suggest checking these coordinates against the territories
                            st.write("\nTo check if a specific point is within any territory, copy these coordinates:")
                            sample_point = unmatched_addresses[0]
                            st.code(f"point = Point({sample_point['lng']}, {sample_point['lat']})")
                            st.write("Then check against each territory:")
                            st.code("""
for name, polygon in territories.items():
    if polygon.contains(point):
        print(f"Point is in {name}")
                            """)
        
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
