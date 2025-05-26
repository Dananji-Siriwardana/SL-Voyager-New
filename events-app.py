from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('events_app.log')
    ]
)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ------------------ Database Initialization ------------------ #

def init_db():
    db_path = os.path.join(os.path.dirname(__file__), 'events.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
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
        conn.commit()
        logging.info(f"Database initialized successfully at {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'events.db')
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
        raise

@app.route('/admin/init-db', methods=['GET'])
def manual_init_db():
    try:
        init_db()
        return jsonify({'message': 'Database initialized successfully'}), 200
    except sqlite3.Error as e:
        logging.error(f"Error in manual database initialization: {e}")
        return jsonify({'error': f"Failed to initialize database: {str(e)}"}), 500

try:
    with app.app_context():
        init_db()
except Exception as e:
    logging.error(f"Failed to initialize database on startup: {e}")
    raise

# ------------------ Admin Panel Routes ------------------ #

@app.route('/admin/events', methods=['GET'])
def admin_panel():
    return render_template('admin_events.html')

@app.route('/api/custom-events', methods=['POST'])
def create_event():
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
def update_event(id):
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
def delete_event(id):
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)