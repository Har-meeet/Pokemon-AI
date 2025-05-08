from pyboy import PyBoy
from abc import ABC, abstractmethod
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.metrics import mean_squared_error
import numpy as np


class Reward(ABC):
    """Abstract base class for all reward types."""
    
    def __init__(self, pyboy: PyBoy):
        self.total_reward = 0
        self.pyboy = pyboy

    @abstractmethod
    def calculate_reward(self, **kwargs):
        """Each subclass must implement this."""
        pass

    def add_reward(self, points):
        """Adds points to the total reward."""
        self.total_reward += points

    def get_total_reward(self):
        """Returns accumulated rewards."""
        return self.total_reward
    
class VisualExplorationReward(Reward):
    def __init__(self, pyboy, screen, mse_threshold=0.005):
        super().__init__(pyboy)
        self.screen = screen
        self.seen_screens = []
        self.mse_threshold = mse_threshold
        self.downsample_shape = (36, 40)

    def capture_downsampled_frame(self):
        frame = self.screen.screen_ndarray()  # shape: (144, 160, 3)
        gray = rgb2gray(frame)                # shape: (144, 160)
        small = resize(gray, self.downsample_shape, anti_aliasing=True)  # shape: (36, 40)
        return small.flatten()

    def is_novel_frame(self, frame_vec):
        if not self.seen_screens:
            return True
        errors = [mean_squared_error(frame_vec, seen) for seen in self.seen_screens]
        min_error = min(errors)
        return min_error > self.mse_threshold

    def calculate_reward(self):
        reward = 0
        frame_vec = self.capture_downsampled_frame()
        if self.is_novel_frame(frame_vec):
            print("Novel screen detected! +10 points")
            self.seen_screens.append(frame_vec)
            reward += 10
        self.add_reward(reward)
        return reward


class ExplorationReward(Reward):
    """Handles movement and NPC interaction rewards."""

    def __init__(self, pyboy: PyBoy):
        super().__init__(pyboy)
        self.visited_positions = {}
        self.visited_npcs = set()
        self.visited_maps = set()
        self.previous_position = None
        self.previous_map = None

    def get_player_position(self):
        x = self.pyboy.get_memory_value(0xD362)
        y = self.pyboy.get_memory_value(0xD361)
        return (x, y)

    def get_map_id(self):
        return self.pyboy.get_memory_value(0xD35E)

    def get_npc_id(self):
        """Simulated function to read NPC ID from memory"""
        return self.pyboy.get_memory_value(0xD400)  # Example address

    def calculate_reward(self):
        reward = 0
        current_position = self.get_player_position()
        current_map = self.get_map_id()

        # Entering a new building
        if current_map not in self.visited_maps:
            print("Entered new building! +100 points")
            reward += 100
            self.visited_maps.add(current_map)

        # New NPC Interaction
        npc_id = self.get_npc_id()
        if npc_id and npc_id not in self.visited_npcs:
            print("New NPC interaction! +100 points")
            self.visited_npcs.add(npc_id)
            reward += 100

        # Backtracking
        if current_position in self.visited_positions:
            if self.visited_positions[current_position] > 3:
                print("Backtracking! -5 points")
                reward -= 5
                self.visited_positions[current_position] -= 1
            else:
                self.visited_positions[current_position] += 1

        # Walking into walls
        if self.previous_position == current_position:
            print("Walking into a wall! -1 point")
            reward -= 3

        # Update tracking variables
        self.previous_position = current_position
        self.previous_map = current_map
        self.add_reward(reward)
        return reward

class BattleReward(Reward):
    """Handles battle-related rewards."""

    def calculate_reward(self, used_move_effectiveness, defeated_wild_pokemon, defeated_trainer_pokemon, won_trainer_battle, trainer_blacked_out, pokemon_fainted):
        reward = 0

        # Reward for attacking
        if used_move_effectiveness == "neutral":
            print("Used neutral move! +5 points")
            reward += 5
        elif used_move_effectiveness == "effective":
            print("Used effective move! +7 points")
            reward += 7
        elif used_move_effectiveness == "ineffective":
            print("Used ineffective move! -2 points")
            reward -= 2

        # Defeating Pokémon
        if defeated_wild_pokemon:
            print("Defeated a wild Pokémon! +7 points")
            reward += 7
        if defeated_trainer_pokemon:
            print("Defeated a trainer Pokémon! +10 points")
            reward += 10
        if won_trainer_battle:
            print("Won a trainer battle! +20 points")
            reward += 20

        # Penalties
        if pokemon_fainted:
            print("Your Pokémon fainted! -10 points")
            reward -= 10
        if trainer_blacked_out:
            print("Trainer blacked out! -20 points")
            reward -= 20

        self.add_reward(reward)
        return reward

class ItemReward(Reward):
    """Handles item collection and money management rewards."""

    def calculate_reward(self, collected_item, money_ran_out):
        reward = 0

        if collected_item:
            print("Collected an item! +5 points")
            reward += 5
        if money_ran_out:
            print("Ran out of money! -15 points")
            reward -= 15

        self.add_reward(reward)
        return reward

class StorylineReward(Reward):
    """Handles major storyline progression rewards."""

    def calculate_reward(self, entered_viridian, entered_pewter, challenged_brock, defeated_brock):
        reward = 0

        if entered_viridian:
            print("Entered Viridian Forest! +30 points")
            reward += 30
        if entered_pewter:
            print("Entered Pewter City! +30 points")
            reward += 30
        if challenged_brock:
            print("Challenged Brock! +50 points")
            reward += 50
        if defeated_brock:
            print("Defeated Brock! +100 points")
            reward += 100

        self.add_reward(reward)
        return reward

class RewardManager:
    """Manages multiple reward systems and aggregates scores."""

    def __init__(self, pyboy: PyBoy):
        self.exploration_reward = ExplorationReward(pyboy)
        self.battle_reward = BattleReward(pyboy)
        self.item_reward = ItemReward(pyboy)
        self.storyline_reward = StorylineReward(pyboy)
        self.visual_explore = VisualExplorationReward(pyboy, pyboy.botsupport_manager().screen())
        self.total_reward = 0

    def calculate_total_reward(self, **kwargs):
        """Calls each reward system and updates the total score."""

        self.total_reward += self.exploration_reward.calculate_reward()
        self.total_reward += self.visual_explore.calculate_reward()

        self.total_reward += self.battle_reward.calculate_reward(
            used_move_effectiveness=kwargs.get("used_move_effectiveness", None),
            defeated_wild_pokemon=kwargs.get("defeated_wild_pokemon", False),
            defeated_trainer_pokemon=kwargs.get("defeated_trainer_pokemon", False),
            won_trainer_battle=kwargs.get("won_trainer_battle", False),
            trainer_blacked_out=kwargs.get("trainer_blacked_out", False),
            pokemon_fainted=kwargs.get("pokemon_fainted", False)
        )
        self.total_reward += self.item_reward.calculate_reward(
            collected_item=kwargs.get("collected_item", False),
            money_ran_out=kwargs.get("money_ran_out", False)
        )
        self.total_reward += self.storyline_reward.calculate_reward(
            entered_viridian=kwargs.get("entered_viridian", False),
            entered_pewter=kwargs.get("entered_pewter", False),
            challenged_brock=kwargs.get("challenged_brock", False),
            defeated_brock=kwargs.get("defeated_brock", False)
        )

        return self.total_reward

    def get_individual_rewards(self):
        return {
            "Exploration": self.exploration_reward.get_total_reward(),
            "Battle": self.battle_reward.get_total_reward(),
            "Item": self.item_reward.get_total_reward(),
            "Storyline": self.storyline_reward.get_total_reward()
        }

    def get_total_reward(self):
        return self.total_reward
