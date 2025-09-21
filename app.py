from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
import requests
import json
from datetime import datetime

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

# Step 3 – Create Enhanced Map
def create_map(df):
    # Calculate the center of all earthquakes for better initial view
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=3,
        tiles='OpenStreetMap'
    )
    
    # Enhanced color palette
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Add cluster statistics to map
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    for _, row in df.iterrows():
        cluster_id = row['cluster']
        color = 'black' if cluster_id == -1 else colors[cluster_id % len(colors)]
        
        # Enhanced popup with more information
        popup_text = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h4 style="color: {color}; margin: 0;">
                {'Outlier' if cluster_id == -1 else f'Cluster {cluster_id}'}
            </h4>
            <hr style="margin: 5px 0;">
            <p><strong>Magnitude:</strong> {row['magnitude']}</p>
            <p><strong>Location:</strong> {row['place']}</p>
            <p><strong>Date:</strong> {row['time'].strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>Coordinates:</strong> {row['latitude']:.3f}, {row['longitude']:.3f}</p>
            <p><strong>Depth:</strong> {row['depth']} km</p>
        </div>
        """
        
        # Size based on magnitude
        radius = min(max(row['magnitude'] * 2, 3), 10)
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add a legend
    legend_html = """
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Cluster Legend</h4>
    """
    
    for i in range(min(8, len(cluster_counts))):
        if i in cluster_counts.index:
            color = colors[i % len(colors)]
            count = cluster_counts[i]
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> Cluster {i} ({count} earthquakes)</p>'
    
    if -1 in cluster_counts.index:
        legend_html += f'<p><i class="fa fa-circle" style="color:black"></i> Outliers ({cluster_counts[-1]} earthquakes)</p>'
    
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_path = "static/map.html"
    m.save(map_path)
    return "static/map.html"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    """API endpoint to get real-time statistics"""
    try:
        df = fetch_earthquake_data()
        df = run_dbscan(df)
        n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
        n_noise = list(df['cluster']).count(-1)
        total_earthquakes = len(df)
        
        return jsonify({
            'total_earthquakes': total_earthquakes,
            'clusters': n_clusters,
            'outliers': n_noise,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/results")
def results():
    df = fetch_earthquake_data()
    df = run_dbscan(df)
    map_file = create_map(df)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    n_noise = list(df['cluster']).count(-1)
    total_earthquakes = len(df)
    
    # Calculate additional statistics
    cluster_stats = df['cluster'].value_counts().sort_index()
    largest_cluster_size = cluster_stats[cluster_stats.index != -1].max() if len(cluster_stats) > 0 else 0
    
    return render_template("results.html", 
                         n_clusters=n_clusters, 
                         n_noise=n_noise,
                         total_earthquakes=total_earthquakes,
                         largest_cluster_size=largest_cluster_size)

if __name__ == "__main__":
    app.run(debug=True)
