"""
Hand Gesture Scribble Game - Flask + Socket.IO Server
"""

import random
import time
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scribble_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────
# WORD BANK
# ─────────────────────────────────────────────
WORD_BANK = [
    "cat", "dog", "house", "tree", "car", "sun", "moon", "star",
    "fish", "bird", "flower", "cloud", "mountain", "river", "boat",
    "plane", "train", "bicycle", "umbrella", "clock", "phone", "book",
    "pizza", "apple", "banana", "guitar", "camera", "hat", "shoe",
    "glasses", "pencil", "candle", "rainbow", "elephant", "butterfly"
]

# ─────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────
game_state = {
    "players": {},          # sid -> {name, score, role}
    "room": "game_room",
    "current_word": None,
    "round_active": False,
    "round_start_time": None,
    "round_duration": 60,   # seconds
    "round_number": 0,
    "drawer_sid": None,
    "guesser_sid": None,
}

def get_other_player(sid):
    for s in game_state["players"]:
        if s != sid:
            return s
    return None

def start_round():
    """Initialize a new round."""
    word = random.choice(WORD_BANK)
    game_state["current_word"] = word
    game_state["round_active"] = True
    game_state["round_start_time"] = time.time()
    game_state["round_number"] += 1

    drawer = game_state["drawer_sid"]
    guesser = game_state["guesser_sid"]

    # Tell the drawer the word
    socketio.emit('round_start', {
        'role': 'drawer',
        'word': word,
        'round': game_state["round_number"],
        'duration': game_state["round_duration"]
    }, to=drawer)

    # Tell the guesser to start guessing
    socketio.emit('round_start', {
        'role': 'guesser',
        'word': None,
        'round': game_state["round_number"],
        'duration': game_state["round_duration"]
    }, to=guesser)

def swap_roles():
    """Swap drawer and guesser."""
    game_state["drawer_sid"], game_state["guesser_sid"] = \
        game_state["guesser_sid"], game_state["drawer_sid"]
    if game_state["drawer_sid"] in game_state["players"]:
        game_state["players"][game_state["drawer_sid"]]["role"] = "drawer"
    if game_state["guesser_sid"] in game_state["players"]:
        game_state["players"][game_state["guesser_sid"]]["role"] = "guesser"

def end_round(winner_sid=None):
    """End the current round."""
    game_state["round_active"] = False

    scores = {
        sid: game_state["players"][sid]["score"]
        for sid in game_state["players"]
    }
    names = {
        sid: game_state["players"][sid]["name"]
        for sid in game_state["players"]
    }

    socketio.emit('round_end', {
        'word': game_state["current_word"],
        'winner': names.get(winner_sid),
        'scores': {names[s]: scores[s] for s in scores}
    }, to=game_state["room"])

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drawer')
def drawer_page():
    return render_template('drawer.html')

@app.route('/guesser')
def guesser_page():
    return render_template('guesser.html')

@app.route('/api/status')
def status():
    return jsonify({
        'players': len(game_state["players"]),
        'round': game_state["round_number"],
        'active': game_state["round_active"]
    })

# ─────────────────────────────────────────────
# SOCKET EVENTS
# ─────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {socketio.server.environ.get('REMOTE_ADDR', 'unknown')} [{socketio.server.manager.get_rooms('/', request_sid())}]")

def request_sid():
    from flask import request
    return request.sid

@socketio.on('join_game')
def handle_join(data):
    from flask import request
    sid = request.sid
    name = data.get('name', f'Player{len(game_state["players"])+1}')

    if len(game_state["players"]) >= 2:
        emit('error', {'message': 'Game is full!'})
        return

    join_room(game_state["room"])

    # Assign role
    if not game_state["players"]:
        role = "drawer"
        game_state["drawer_sid"] = sid
    else:
        role = "guesser"
        game_state["guesser_sid"] = sid

    game_state["players"][sid] = {
        'name': name,
        'score': 0,
        'role': role
    }

    emit('joined', {
        'name': name,
        'role': role,
        'player_count': len(game_state["players"])
    })

    # Notify other player
    other = get_other_player(sid)
    if other:
        socketio.emit('player_joined', {
            'name': name,
            'player_count': len(game_state["players"])
        }, to=other)

    # Start game when 2 players joined
    if len(game_state["players"]) == 2:
        socketio.emit('game_ready', {
            'message': 'Both players connected! Starting in 3 seconds...'
        }, to=game_state["room"])
        socketio.sleep(3)
        start_round()

@socketio.on('draw_data')
def handle_draw(data):
    """Relay drawing coordinates to guesser."""
    from flask import request
    sid = request.sid
    if game_state["players"].get(sid, {}).get("role") == "drawer":
        other = get_other_player(sid)
        if other:
            socketio.emit('draw_update', data, to=other)

@socketio.on('clear_canvas')
def handle_clear():
    from flask import request
    sid = request.sid
    if game_state["players"].get(sid, {}).get("role") == "drawer":
        other = get_other_player(sid)
        if other:
            socketio.emit('canvas_cleared', {}, to=other)

@socketio.on('submit_guess')
def handle_guess(data):
    from flask import request
    sid = request.sid
    guess = data.get('guess', '').strip().lower()
    correct_word = (game_state["current_word"] or "").lower()

    if not game_state["round_active"]:
        return

    # Broadcast guess to both players
    name = game_state["players"].get(sid, {}).get("name", "?")
    socketio.emit('guess_broadcast', {
        'name': name,
        'guess': guess
    }, to=game_state["room"])

    if guess == correct_word:
        # Only the guesser can score and end the round
        if game_state["players"].get(sid, {}).get("role") == "guesser":
            game_state["players"][sid]["score"] += 1
            end_round(winner_sid=sid)

            # Schedule next round
            def next_round():
                socketio.sleep(4)
                if len(game_state["players"]) == 2:
                    swap_roles()
                    socketio.emit('canvas_cleared', {}, to=game_state["room"])
                    start_round()

            socketio.start_background_task(next_round)

@socketio.on('time_up')
def handle_time_up():
    if game_state["round_active"]:
        end_round(winner_sid=None)

        def next_round():
            socketio.sleep(4)
            if len(game_state["players"]) == 2:
                swap_roles()
                socketio.emit('canvas_cleared', {}, to=game_state["room"])
                start_round()

        socketio.start_background_task(next_round)

@socketio.on('disconnect')
def handle_disconnect():
    from flask import request
    sid = request.sid
    if sid in game_state["players"]:
        name = game_state["players"][sid]["name"]
        del game_state["players"][sid]
        game_state["round_active"] = False
        game_state["drawer_sid"] = None
        game_state["guesser_sid"] = None
        game_state["round_number"] = 0
        socketio.emit('player_left', {
            'name': name,
            'message': f'{name} disconnected. Waiting for players...'
        }, to=game_state["room"])

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print(f"  Scribble Game Server Starting on port {port}...")
    print("  Open the URL provided by your hosting service")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
