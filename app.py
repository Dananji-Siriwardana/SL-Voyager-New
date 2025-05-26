from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import aiohttp
import asyncio
import sqlite3
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import ast

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'e99d52c0eb990dcfcd77d222be9812199a528d9ad1ead2403c52f2b4f9470d11'
CORS(app)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('events_app.log')
    ]
)

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return User(user['id'], user['username'], user['email']) if user else None

# Database connection helper
def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'events.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
            -- is_admin BOOLEAN DEFAULT 0  -- Uncomment to add is_admin field
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            attraction_name TEXT,
            rating INTEGER,
            visit_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            start_date TEXT NOT NULL,
            type TEXT NOT NULL,
            target_audience TEXT NOT NULL,
            image_url TEXT,
            description TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_cache (
            lat REAL,
            lon REAL,
            timestamp TEXT,
            temperature REAL,
            description TEXT,
            PRIMARY KEY (lat, lon)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS venue_coordinates (
            address TEXT PRIMARY KEY,
            lat REAL,
            lng REAL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully")

# Call init_db on startup
with app.app_context():
    init_db()

# --- Attraction Recommendation System ---

# Load model and encoders
model_data = joblib.load('model_data.pkl')
rf_model = model_data['rf_model']
knn_model = model_data['knn_model']
traveler_mlb = model_data['traveler_mlb']
activity_mlb = model_data['activity_mlb']
season_mlb = model_data['season_mlb']

# Load attraction data
data = pd.read_csv('Places.csv')
data['Traveler Type'] = data['Traveler Type'].apply(ast.literal_eval)
data['Activity Type'] = data['Activity Type'].apply(ast.literal_eval)
data['Seasons'] = data['Seasons'].apply(ast.literal_eval)
data['Best Months'] = data['Best Months'].apply(ast.literal_eval)

# Weather API key
WEATHER_API_KEY = '714502160807c4d7a00552387f3748f7'

# Season detection
def get_current_season(month=None):
    if month is None:
        month = datetime.now().month
    if month in [1, 2, 3]:
        return 'Dry Season'
    elif month in [4, 5]:
        return 'First Inter-Monsoon'
    elif month in [6, 7, 8, 9]:
        return 'Southwest Monsoon'
    else:
        return 'Second Inter-Monsoon'

# Asynchronous weather fetch
async def get_weather_async(lat, lon, session):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT temperature, description FROM weather_cache
            WHERE lat = ? AND lon = ? AND timestamp > ?
        ''', (lat, lon, (datetime.now() - pd.Timedelta(hours=1)).isoformat()))
        cached = cursor.fetchone()
        if cached:
            conn.close()
            return {'temperature': cached['temperature'], 'description': cached['description']}

        url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric'
        async with session.get(url) as response:
            weather_data = await response.json()
            temp = weather_data['main']['temp']
            desc = weather_data['weather'][0]['description']
            cursor.execute('''
                INSERT OR REPLACE INTO weather_cache (lat, lon, timestamp, temperature, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (lat, lon, datetime.now().isoformat(), temp, desc))
            conn.commit()
            conn.close()
            return {'temperature': temp, 'description': desc}
    except Exception as e:
        logging.error(f"Error fetching weather for {lat},{lon}: {e}")
        return {'temperature': 'N/A', 'description': 'Weather data unavailable'}

# User profile helper
def get_user_profile(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT attraction_name, rating FROM user_activities WHERE user_id = ?', (user_id,))
    activities = cursor.fetchall()
    conn.close()
    if not activities:
        return np.zeros(len(traveler_mlb.classes_) + len(activity_mlb.classes_))
    profile = np.zeros(len(traveler_mlb.classes_) + len(activity_mlb.classes_))
    for act in activities:
        attr = data[data['Name'] == act['attraction_name']]
        if not attr.empty:
            attr = attr.iloc[0]
            attr_vec = np.concatenate([
                traveler_mlb.transform([attr['Traveler Type']])[0],
                activity_mlb.transform([attr['Activity Type']])[0]
            ])
            profile += attr_vec * (act['rating'] / 5.0)
    return profile / len(activities) if activities else profile

# Admin check
def is_admin(user):
    return user.username == 'admin'  # Replace with user.is_admin if you add is_admin field

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        password_hash = generate_password_hash(password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                           (username, email, password_hash))
            conn.commit()
            conn.close()
            logging.info(f"User registered: {username}")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            logging.error("Username or email already exists")
            return render_template('signup.html', error="Username or email already exists", active_page='register')
    return render_template('signup.html', active_page='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            login_user(User(user['id'], user['username'], user['email']))
            logging.info(f"User logged in: {username}")
            return redirect(url_for('home'))
        logging.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html', error="Invalid credentials", active_page='login')
    return render_template('login.html', active_page='login')

@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logging.info(f"User logged out: {username}")
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT attraction_name, rating, visit_date FROM user_activities WHERE user_id = ?', (current_user.id,))
    activities = cursor.fetchall()
    conn.close()
    return render_template('profile.html', user=current_user, activities=activities, active_page='profile')

# Attraction Recommendation Routes
# Update the /options route to include file existence checks
@app.route('/options', methods=['GET'])
@login_required
def get_options():
    try:
        if not os.path.exists('model_data.pkl'):
            logging.error("model_data.pkl file is missing")
            return jsonify({'error': 'Model data not found', 'traveler_types': [], 'activity_types': []}), 500
        if not os.path.exists('Places.csv'):
            logging.error("Places.csv file is missing")
            return jsonify({'error': 'Places data not found', 'traveler_types': [], 'activity_types': []}), 500
        traveler_types = sorted(traveler_mlb.classes_.tolist())
        activity_types = sorted(activity_mlb.classes_.tolist())
        logging.info(f"Traveler Types: {traveler_types}")
        logging.info(f"Activity Types: {activity_types}")
        return jsonify({
            'traveler_types': traveler_types,
            'activity_types': activity_types
        })
    except Exception as e:
        logging.error(f"Error in get_options: {str(e)}", exc_info=True)
        return jsonify({'error': f"Server error: {str(e)}", 'traveler_types': [], 'activity_types': []}), 500

# Update the /recommends route to log more details
@app.route('/recommends', methods=['POST'])
@login_required
async def recommend():
    try:
        user_data = request.get_json()
        traveler_types = user_data.get('traveler_types', [])
        activity_types = user_data.get('activity_types', [])
        user_id = current_user.id
        logging.info(f"User {user_id} Input: traveler_types={traveler_types}, activity_types={activity_types}")

        if not traveler_types or not activity_types:
            logging.warning("Empty traveler or activity types")
            return jsonify({'error': 'Select at least one traveler type and activity type'}), 400

        current_month = datetime.now().month
        current_season = get_current_season()
        logging.info(f"Current Month: {current_month}, Season: {current_season}")

        tt_vec = traveler_mlb.transform([traveler_types])[0]
        at_vec = activity_mlb.transform([activity_types])[0]
        month_vec = np.zeros(12)
        month_vec[current_month-1] = 1
        season_vec = season_mlb.transform([[current_season]])[0]
        user_vec = np.concatenate([tt_vec, at_vec, month_vec, season_vec])

        user_profile = get_user_profile(user_id)
        logging.info(f"User Profile Shape: {user_profile.shape}")

        filtered_data = data[
            data['Seasons'].apply(lambda x: current_season in x or not x) |
            data['Traveler Type'].apply(lambda x: any(t in x for t in traveler_types) or not x) |
            data['Activity Type'].apply(lambda x: any(a in x for a in activity_types) or not x)
        ]
        logging.info(f"Filtered {len(filtered_data)} attractions")

        if filtered_data.empty:
            logging.warning("No attractions after filtering")
            return jsonify({'month': current_month, 'season': current_season, 'recommendations': []}), 200

        rf_inputs = []
        attr_profiles = []
        valid_attractions = []
        for _, attr in filtered_data.iterrows():
            try:
                attr_vec = np.concatenate([
                    traveler_mlb.transform([attr['Traveler Type']])[0],
                    activity_mlb.transform([attr['Activity Type']])[0],
                    season_mlb.transform([attr['Seasons']])[0]
                ])
                rf_inputs.append(np.concatenate([user_vec, attr_vec]))
                attr_profiles.append(np.concatenate([
                    traveler_mlb.transform([attr['Traveler Type']])[0],
                    activity_mlb.transform([attr['Activity Type']])[0]
                ]))
                valid_attractions.append(attr)
            except Exception as e:
                logging.error(f"Error processing attraction {attr['Name']}: {e}")
                continue

        if not valid_attractions:
            logging.warning("No valid attractions after processing")
            return jsonify({'month': current_month, 'season': current_season, 'recommendations': []}), 200

        rf_scores = rf_model.predict(rf_inputs)
        knn_probs = knn_model.predict_proba([user_vec])[0]
        logging.info(f"RF Scores Shape: {rf_scores.shape}, KNN Probs: {knn_probs.shape}")

        recommendations = []
        async with aiohttp.ClientSession() as session:
            for i, (attr, rf_score, attr_profile) in enumerate(zip(valid_attractions, rf_scores, attr_profiles)):
                similarity = cosine_similarity([user_profile], [attr_profile])[0][0]
                final_score = 0.5 * rf_score + 0.2 * max(knn_probs) + 0.3 * similarity
                weather = await get_weather_async(attr['Latitude'], attr['Longitude'], session)
                recommendations.append({
                    'name': attr['Name'],
                    'score': float(final_score),
                    'weather': weather,
                    'location': {'lat': attr['Latitude'], 'lon': attr['Longitude']},
                    'description': attr['Description']
                })

        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:5]
        logging.info(f"Generated {len(recommendations)} recommendations")
        return jsonify({
            'month': current_month,
            'season': current_season,
            'recommendations': recommendations
        })
    except Exception as e:
        logging.error(f"Error in recommend: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/traveler', methods=['GET'])
@login_required
def traveler():
    return render_template('traveler.html', active_page='traveler')

@app.route('/activity', methods=['GET'])
@login_required
def activity():
    return render_template('activity.html', active_page='activity')

@app.route('/recommendations', methods=['GET'])
@login_required
def recommendations():
    return render_template('recommendations.html', active_page='recommendations')

# Hotel Recommendation Routes
model = joblib.load('event_recommendation_model.pkl')
dataset = pd.read_csv('Dataset.csv')

async def get_coordinates_async(place, session):
    API_KEY = 'AIzaSyA2MXdyzbpEbtQZxVLdBFQUg9qO_3ASknI'
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT lat, lng FROM venue_coordinates
        WHERE address = ?
    ''', (place,))
    cached = cursor.fetchone()
    if cached:
        conn.close()
        return {'lat': cached['lat'], 'lng': cached['lng']}

    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={API_KEY}'
    try:
        async with session.get(url) as response:
            data = await response.json()
            if data['status'] == 'OK':
                location = data['results'][0]['geometry']['location']
                cursor.execute('''
                    INSERT INTO venue_coordinates (address, lat, lng)
                    VALUES (?, ?, ?)
                ''', (place, location['lat'], location['lng']))
                conn.commit()
                conn.close()
                return {'lat': location['lat'], 'lng': location['lng']}
            else:
                logging.error(f"Geocoding error for {place}: {data['status']}")
                return None
    except Exception as e:
        logging.error(f"Error in get_coordinates: {e}")
        return None

def recommend_venue(event_type, guest_count, min_budget, max_budget, special_requirements=None):
    special_requirements = special_requirements if special_requirements else "None"
    input_data = pd.DataFrame({
        'Event_Type': [event_type],
        'Guest_Count': [guest_count],
        'Min_Budget': [min_budget],
        'Max_Budget': [max_budget],
        'Special_Requirements': [special_requirements]
    })
    probabilities = model.predict_proba(input_data)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_venues = model.named_steps['classifier'].classes_[top_3_indices]
    top_3_probabilities = probabilities[top_3_indices]
    recommendations = []
    for venue, prob in zip(top_3_venues, top_3_probabilities):
        venue_details = dataset[dataset['Venue_Name'] == venue].iloc[0].to_dict()
        recommendations.append({
            'name': venue,
            'probability': prob,
            'details': venue_details
        })
    return recommendations

@app.route("/recommended", methods=["GET", "POST"])
@login_required
async def index():
    recommendations = []
    map_locations = []
    form_data = None
    error = None

    if request.method == "POST":
        try:
            event_type = request.form.get("event_type")
            guest_count = int(request.form.get("guest_count"))
            min_budget = int(request.form.get("min_budget"))
            max_budget = int(request.form.get("max_budget"))
            special_requirements = request.form.get("special_requirements")
            recommendations = recommend_venue(event_type, guest_count, min_budget, max_budget, special_requirements)
            async with aiohttp.ClientSession() as session:
                for venue in recommendations:
                    address = venue['details'].get('Location', '')
                    coordinates = await get_coordinates_async(address, session)
                    if coordinates:
                        map_locations.append({
                            'name': venue['name'],
                            'address': address,
                            'type': venue['details'].get('Venue_Type', ''),
                            'rating': venue['details'].get('User_Rating', ''),
                            'probability': venue['probability'],
                            'lat': coordinates['lat'],
                            'lng': coordinates['lng']
                        })
            form_data = request.form
        except Exception as e:
            logging.error(f"Error in recommended: {e}", exc_info=True)
            error = str(e)

    return render_template("index.html",
                           recommendations=recommendations,
                           map_locations=map_locations,
                           form_data=form_data,
                           error=error,
                           active_page='recommended')

# Admin Panel Routes
@app.route('/admin/events', methods=['GET'])
@login_required
def admin_panel():
    if not is_admin(current_user):
        logging.warning(f"Unauthorized access to admin panel by {current_user.username}")
        return redirect(url_for('home'))
    return render_template('admin-events.html', active_page='admin_events')

@app.route('/api/custom-events', methods=['POST'])
@login_required
def create_event():
    if not is_admin(current_user):
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        data = request.json
        title = data.get('title')
        start_date = data.get('start_date')
        event_type = data.get('type')
        target_audience = data.get('target_audience')
        image_url = data.get('image_url')
        description = data.get('description')

        if not all([title, start_date, event_type, target_audience]):
            return jsonify({'error': 'Missing required fields'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO custom_events (title, start_date, type, target_audience, image_url, description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, start_date, event_type, target_audience, image_url, description))
        conn.commit()
        conn.close()
        logging.info(f"Event created: {title}")
        return jsonify({'message': 'Event created successfully'}), 201
    except sqlite3.Error as e:
        logging.error(f"Error creating event: {e}")
        return jsonify({'error': f"Database error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error creating event: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/custom-events', methods=['GET'])
@login_required
def get_events():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM custom_events WHERE start_date >= ?', (datetime.now().strftime('%Y-%m-%d'),))
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        logging.info(f"Fetched {len(events)} events")
        return jsonify(events), 200
    except sqlite3.Error as e:
        logging.error(f"Error fetching events: {e}")
        return jsonify({'error': f"Database error: {str(e)}"}), 500

@app.route('/api/custom-events/<int:id>', methods=['PUT'])
@login_required
def update_event(id):
    if not is_admin(current_user):
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        data = request.json
        title = data.get('title')
        start_date = data.get('start_date')
        event_type = data.get('type')
        target_audience = data.get('target_audience')
        image_url = data.get('image_url')
        description = data.get('description')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE custom_events
            SET title = ?, start_date = ?, type = ?, target_audience = ?, image_url = ?, description = ?
            WHERE id = ?
        ''', (title, start_date, event_type, target_audience, image_url, description, id))
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Event not found'}), 404
        logging.info(f"Event updated: ID {id}")
        return jsonify({'message': 'Event updated successfully'}), 200
    except sqlite3.Error as e:
        logging.error(f"Error updating event: {e}")
        return jsonify({'error': f"Database error: {str(e)}"}), 500

@app.route('/api/custom-events/<int:id>', methods=['DELETE'])
@login_required
def delete_event(id):
    if not is_admin(current_user):
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM custom_events WHERE id = ?', (id,))
        conn.commit()
        conn.close()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Event not found'}), 404
        logging.info(f"Event deleted: ID {id}")
        return jsonify({'message': 'Event deleted successfully'}), 200
    except sqlite3.Error as e:
        logging.error(f"Error deleting event: {e}")
        return jsonify({'error': f"Database error: {str(e)}"}), 500
    
# --- Forecast ---#

# New route to check authentication status
@app.route('/check_auth')
def check_auth():
    return jsonify({"authenticated": current_user.is_authenticated})

# New route for forecasting page
@app.route('/forecast')
@login_required
def forecast():
    return render_template('forecast.html', streamlit_url='http://localhost:8501')



# Homepage route
@app.route("/", methods=["GET"])
def home():
    current_month = datetime.now().month
    current_season = get_current_season()
    return render_template('homepage.html', month=current_month, season=current_season, active_page='home')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)