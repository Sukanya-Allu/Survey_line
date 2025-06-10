import streamlit as st
import math
import numpy as np
import xml.etree.ElementTree as ET
import pyproj
import tempfile
import simplekml
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union, transform as proj_transform
from shapely.affinity import rotate

st.set_page_config(page_title="Survey Line Generator", layout="wide")
st.title("📍 Drone Survey Line Generator")

# Load polygon from uploaded KML
def load_polygon(kml_file):
    ns = "{http://www.opengis.net/kml/2.2}"
    tree = ET.parse(kml_file)
    root = tree.getroot()
    parts = []
    for coord in root.findall(f".//{ns}Polygon/{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates"):
        pts = []
        for tok in (coord.text or "").split():
            lon, lat = tok.split(",")[:2]
            pts.append((float(lon), float(lat)))
        if len(pts) >= 3:
            parts.append(Polygon(pts))
    return unary_union(parts)

def project_to_utm(poly):
    lon, lat = poly.centroid.x, poly.centroid.y
    zone = int((lon + 180)//6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    inv = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    return proj_transform(fwd, poly), inv

def get_polygon_edges(poly_m):
    mrr = poly_m.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    edges = [(coords[i], coords[(i+1)%4]) for i in range(4)]
    lengths = [math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in edges]
    sorted_edges = sorted(zip(edges, lengths), key=lambda t: t[1], reverse=True)
    e1 = sorted_edges[0][0]
    e2 = sorted_edges[1][0]
    y1 = (e1[0][1] + e1[1][1]) / 2
    y2 = (e2[0][1] + e2[1][1]) / 2
    return LineString(e1) if y1 < y2 else LineString(e2)

def make_survey_lines(poly_m, spacing, tolerance, turn_radius):
    bottom = get_polygon_edges(poly_m)
    dx = bottom.coords[1][0] - bottom.coords[0][0]
    dy = bottom.coords[1][1] - bottom.coords[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    rot = rotate(poly_m, -angle, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rot.bounds
    ys = np.arange(miny + spacing/2, maxy + spacing + 1e-6, spacing)
    path = []
    for idx, y in enumerate(ys):
        base = LineString([(minx - spacing, y), (maxx + spacing, y)])
        inter = base.intersection(rot)
        coords = []
        if inter.is_empty:
            continue
        if isinstance(inter, LineString):
            coords = list(inter.coords)
        elif isinstance(inter, MultiLineString):
            longest = max(inter.geoms, key=lambda s: s.length)
            coords = list(longest.coords)
        if not coords:
            continue
        coords.sort(key=lambda p: p[0])
        start_x, end_x = coords[0][0], coords[-1][0]
        strip = [(start_x, y), (end_x, y)] if idx % 2 == 0 else [(end_x, y), (start_x, y)]
        if idx == 0:
            path.extend(strip)
        else:
            px, py = path[-1]
            nx, ny = strip[0]
            if turn_radius == 0:
                path.append((nx, ny))
            else:
                turn_x = px + turn_radius if idx % 2 == 1 else px - turn_radius
                path.extend([(turn_x, py), (turn_x, ny), (nx, ny)])
        path.extend(strip[1:])
    snake = LineString(path)
    return [rotate(snake, angle, origin=poly_m.centroid, use_radians=False)]

def estimate_path_length(poly_m, spacing, turn_radius):
    bottom = get_polygon_edges(poly_m)
    dx = bottom.coords[1][0] - bottom.coords[0][0]
    dy = bottom.coords[1][1] - bottom.coords[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    rot = rotate(poly_m, -angle, origin='centroid', use_radians=False)
    minx, miny, maxx, maxy = rot.bounds
    n = max(math.ceil((maxy - miny) / spacing), 1)
    L = maxx - minx
    T = spacing * (n - 1)
    return n * L + T, n

def write_kml(poly_geo, lines_geo, altitude):
    k = simplekml.Kml()
    poly_coords = [(x, y, 0) for x, y in poly_geo.exterior.coords]
    k.newpolygon(name="Survey Area", outerboundaryis=poly_coords)
    for ln in lines_geo:
        pts = [(x, y, altitude + 10) for x, y in ln.coords]
        ls = k.newlinestring(name="Survey Path", coords=pts)
        ls.altitudemode = simplekml.AltitudeMode.absolute
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
    k.save(tmp_file.name)
    return tmp_file.name

# Sidebar form
with st.sidebar:
    st.header("Input Parameters")
    kml_file = st.file_uploader("Upload Polygon KML", type="kml")
    altitude = st.number_input("Flight Altitude (m)", value=320.0)
    focal_length = st.number_input("Camera Focal Length (mm)", value=35.0)
    sensor_width = st.number_input("Sensor Width (mm)", value=35.7)
    sensor_height = st.number_input("Sensor Height (mm)", value=23.8)
    image_width = st.number_input("Image Width (px)", value=9504)
    image_height = st.number_input("Image Height (px)", value=6336)
    side_overlap = st.number_input("Side Overlap (%)", min_value=0.0, max_value=100.0, value=60.0) / 100
    front_overlap = st.number_input("Front Overlap (%)", min_value=0.0, max_value=100.0, value=80.0) / 100
    drone_speed = st.number_input("Drone Speed (m/s)", value=20.0)
    turn_radius = st.number_input("Turn Radius (m)", value=300.0)

if kml_file:
    poly_geo = load_polygon(kml_file)
    poly_m, to_geo = project_to_utm(poly_geo)
    fw = altitude * (sensor_width / focal_length)
    fl = altitude * (sensor_height / focal_length)
    spacing = fw * (1 - side_overlap)
    trigger_dist = fl * (1 - front_overlap)
    gsd = (sensor_width * altitude * 100) / (focal_length * image_width)
    survey_area = poly_m.area / 1e6
    est_len, est_n = estimate_path_length(poly_m, spacing, turn_radius)
    lines_m = make_survey_lines(poly_m, spacing, 10.0, turn_radius)
    total_len = sum(ln.length for ln in lines_m)
    photo_count = int(total_len / trigger_dist) + 1
    interval = trigger_dist / drone_speed
    lines_geo = [proj_transform(to_geo, ln) for ln in lines_m]
    out_kml = write_kml(poly_geo, lines_geo, altitude)

    st.success("Survey lines generated.")
    st.download_button("📥 Download Survey KML", open(out_kml, "rb"), file_name="survey_path.kml")
    
    st.subheader("📊 Results")
    st.write(f"**GSD**: {gsd:.2f} cm/px")
    st.write(f"**Trigger Distance**: {trigger_dist:.2f} m")
    st.write(f"**Survey Area**: {survey_area:.2f} km²")
    st.write(f"**Estimated Strips**: {est_n}")
    st.write(f"**Photo Count**: {photo_count}")
    st.write(f"**Interval**: {interval:.2f} s")
