#dqn_agent.py
import numpy as np
import random
import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import json

# AI Definitions
MEM_MAX_SIZE = 100000 
GAMMA = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

MODEL_DIR = 'snake_models_v2' 
MODEL_NAME = 'dqn_snake_model_v2.keras'
METADATA_NAME = 'training_metadata.json'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_NAME)

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEM_MAX_SIZE)
        self.gamma = GAMMA    
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        
        # Training optimization
        self.training_step = 0
        self.update_frequency = 4  # Train every N steps
        
        # Training statistics
        self.total_episodes = 0
        self.total_steps = 0
        
        # Create model directory if it doesn't exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            print(f"Created model directory: {MODEL_DIR}")
        
        # Build or load model
        self.model = self._build_model()
        
        # Load training metadata (epsilon, episodes, etc.)
        self._load_metadata()

    def _build_model(self):
        """Build or load existing model."""
        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            try:
                print(f"\n🔄 Loading existing model from: {MODEL_PATH}")
                model = tf.keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model architecture: {self.state_size} -> 256 -> 256 -> 128 -> 64 -> {self.action_size}")
                
                # Verify model architecture matches expected state size
                expected_input_shape = (None, self.state_size)
                actual_input_shape = model.input_shape
                
                if actual_input_shape != expected_input_shape:
                    print(f"⚠️ Warning: Model input shape {actual_input_shape} doesn't match expected {expected_input_shape}")
                    print("Creating new model with correct architecture...")
                    return self._create_new_model()
                
                return model
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("Creating new model instead...")
                return self._create_new_model()
        else:
            print(f"\n📝 No existing model found at: {MODEL_PATH}")
            return self._create_new_model()

    def _create_new_model(self):
        """Create a new DQN model."""
        print(f"🆕 Creating new DQN V2 model with {self.state_size} input states.")
        
        model = Sequential()
        
        # Deeper network for complex spatial patterns
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))  # Prevent overfitting
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        
        model.add(Dense(64, activation='relu'))
        
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        print(f"📊 Model architecture: {self.state_size} -> 256 -> 256 -> 128 -> 64 -> {self.action_size}")
        return model

    def _load_metadata(self):
        """Load training metadata (epsilon, episodes, steps)."""
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                
                self.epsilon = metadata.get('epsilon', 1.0)
                self.total_episodes = metadata.get('total_episodes', 0)
                self.total_steps = metadata.get('total_steps', 0)
                
                print(f"📈 Training metadata loaded:")
                print(f"   ├─ Epsilon: {self.epsilon:.4f}")
                print(f"   ├─ Total Episodes: {self.total_episodes}")
                print(f"   └─ Total Steps: {self.total_steps}")
                
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")
                print("Starting with default values...")
        else:
            print(f"📝 No metadata found. Starting fresh training session.")

    def _save_metadata(self):
        """Save training metadata (epsilon, episodes, steps)."""
        metadata = {
            'epsilon': float(self.epsilon),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate
        }
        
        try:
            with open(METADATA_PATH, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            ahora = datetime.datetime.now()
            print(f"[{ahora}] SE ESTA EXPLORANDO") 
            return random.randrange(self.action_size)
        
        state = np.array(state).reshape(1, self.state_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return
        
        # Only train every N steps to reduce overhead
        self.training_step += 1
        if self.training_step % self.update_frequency != 0:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        # Prepare batch arrays for vectorized operations
        states = np.zeros((BATCH_SIZE, self.state_size))
        targets = np.zeros((BATCH_SIZE, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            target = reward
            if not done:
                next_state_reshaped = np.array(next_state).reshape(1, self.state_size)
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state_reshaped, verbose=0)[0]
                )
            
            target_f = self.model.predict(states[i].reshape(1, self.state_size), verbose=0)
            target_f[0][action] = target
            targets[i] = target_f[0]
        
        # Single batch training call
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track total steps
        self.total_steps += 1

    def on_episode_end(self):
        """Call this at the end of each episode."""
        self.total_episodes += 1

    def save_model(self):
        """Save model and metadata to disk."""
        try:
            self.model.save(MODEL_PATH)
            self._save_metadata()
            print(f"\n💾 Model and metadata saved successfully!")
            print(f"   ├─ Model: {MODEL_PATH}")
            print(f"   ├─ Metadata: {METADATA_PATH}")
            print(f"   ├─ Epsilon: {self.epsilon:.4f}")
            print(f"   └─ Total Episodes: {self.total_episodes}")
        except Exception as e:
            print(f"\n❌ Error saving model: {e}")


