import numpy as np
import matplotlib.pyplot as plt
from pyboy import PyBoy, WindowEvent 
from pathlib import Path

from Reward_System import RewardManager 
from image_processing.Dialogue_Classification import Dialogue_Classification
from image_processing.Dialogue_Box_detection import dialogue_box_code

class Emu:
    def __init__(self, game_path, state_path, window_type="SDL2", ticks=[50, 50]):
        self.pyboy = PyBoy(game_path, debugging=False, disable_input=False, window_type=window_type)
        self.ticks_per_actions = ticks[0]
        self.wait_ticks = ticks[1]
        with open(state_path, "rb") as f:
            self.pyboy.load_state(f)
        
        self.pyboy.set_emulation_speed(6)
        
        self.RewardManager = RewardManager(self.pyboy)

        self.actions = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT
        ]

        self.actions_names = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "B",
            "START",
            "SELECT"
        ]

    def get_release_event(self, action):
        release_mapping = {
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT: WindowEvent.RELEASE_BUTTON_SELECT
        }
        return release_mapping.get(action, None)
    
    def in_dialogue_box(self, turn):

        # Capture the bottom 24 pixels (dialogue region)
        screen = self.pyboy.botsupport_manager().screen().screen_ndarray()

        dialogue_box_code_hash = dialogue_box_code(screen, turn)
        if dialogue_box_code_hash == 0:
            return False
        
        if dialogue_box_code_hash == 1:
            return True
        
        self.RewardManager.exploration_reward.visited_dialogues.add(dialogue_box_code_hash)

        return True
    
    def check_state(self):
        i = 0
        while self.in_dialogue_box(i):
            print("In dialogue... pressing A")
            self.press_button(WindowEvent.PRESS_BUTTON_A)
            i += 1

    def get_vals(self):
        x = self.pyboy.get_memory_value(0xD362)
        y = self.pyboy.get_memory_value(0xD361)
        hp = self.pyboy.get_memory_value(0xD16D)
        money = self.pyboy.get_memory_value(0xD347)
        return np.array([x, y, hp, money], dtype=np.float32)

    def press_button(self, action):
        self.pyboy.send_input(action)
        for i in range(self.ticks_per_actions):
            self.pyboy.tick()
            if i == 8:
                self.pyboy.send_input(self.get_release_event(action))
        for _ in range(self.wait_ticks):
            self.pyboy.tick()

    def step(self, action_idx):
        action = self.actions[action_idx]
        self.press_button(action)

        reward = self.RewardManager.calculate_total_reward()
        done = self.pyboy.get_memory_value(0xD22E) == 1
        vals = self.get_vals()
        print(f"State: X={vals[0]}, Y={vals[1]}, HP={vals[2]}, Money={vals[3]}, Reward={reward}")

        return vals, reward, done

    def get_valid_action_indices(self, state):
        """Define your condition logic here to mask buttons like B, START, SELECT."""
        allow_b = False      # Replace with logic like: `not in_menu(state)`
        allow_start = False # Replace with logic like: `not in_battle(state)`
        allow_select = False

        valid_indices = []
        for idx, name in enumerate(self.actions_names):
            if name == "B" and not allow_b:
                continue
            if name == "START" and not allow_start:
                continue
            if name == "SELECT" and not allow_select:
                continue
            valid_indices.append(idx)

        return valid_indices

    def reset(self):
        with open("Pokemon-Game\\Starting.state", "rb") as f:
            self.pyboy.load_state(f)
        return self.get_vals()

    def close(self):
        self.pyboy.stop()

    def save_screen(self, filename):

        screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        
        # Create the screenshots directory if it doesn't exist
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        # Save the screen
        filepath = screenshots_dir / filename
        plt.imsave(filepath, screen)
        
        print(f"Screen saved as {filepath}")
