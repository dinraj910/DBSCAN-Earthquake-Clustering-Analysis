from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
import requests
import json

app = Flask(__name__)

# Step 1 – Fetch data from USGS
def fetch_earthquake_data():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2022-04-01&endtime=2025-01-01&minmagnitude=4.5"
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data['features'])
    df = df[['properties.time', 'properties.mag', 'properties.place', 'geometry.coordinates']]
    df.rename(columns={
        'properties.time': 'time',
        'properties.mag': 'magnitude',
        'properties.place': 'place'
    }, inplace=True)
    df[['longitude', 'latitude', 'depth']] = pd.DataFrame(df['geometry.coordinates'].tolist(), index=df.index)
    df.drop(columns=['geometry.coordinates'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# Step 2 – Run DBSCAN
def run_dbscan(df):
    X = df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    db = DBSCAN(eps=0.2, min_samples=5)
    labels = db.fit_predict(X_scaled)
    df['cluster'] = labels
    return df

# Step 3 – Create Map
def create_map(df):
    m = folium.Map(location=[0,0], zoom_start=2)
    colors = ['red','blue','green','purple','orange','darkred','lightred',
              'beige','darkblue','darkgreen','cadetblue','darkpurple','white',
              'pink','lightblue','lightgreen','gray','black','lightgray']
    for _, row in df.iterrows():
        cluster_id = row['cluster']
        color = 'black' if cluster_id == -1 else colors[cluster_id % len(colors)]
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"Magnitude: {row['magnitude']}, Place: {row['place']}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
    map_path = "templates/map.html"
    m.save(map_path)
    return "map.html"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    df = fetch_earthquake_data()
    df = run_dbscan(df)
    map_file = create_map(df)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    n_noise = list(df['cluster']).count(-1)
    return render_template("results.html", n_clusters=n_clusters, n_noise=n_noise, map_file=map_file)

if __name__ == "__main__":
    app.run(debug=True)
