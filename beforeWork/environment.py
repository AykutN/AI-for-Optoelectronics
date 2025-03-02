import numpy as np
import dataloader as dl

class Env():
    def __init__(self, data):
        self.data = data
        self.result = data
        self.d1p = 50
        self.d2p = 50
        self.new_location = self._get_location(self.d1p, self.d2p)
        self.current_step = 0
        self.max_steps = 500  # Epizod başına maksimum adım
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)
        self.highest_avt = self.new_location

    def _generate_action_space(self):
        actions = ["forward", "backward", "hold"]
        action_space = []
        for action1 in actions:
            for action2 in actions:
                action_space.append([action1, action2])
        return action_space

    def _get_location(self, d1, d2):
        location = self.result[(self.result['d1'] == d1) & (self.result['d2'] == d2)]
        if location.empty:
            raise ValueError(f"Invalid position: d1={d1}, d2={d2}")
        return location['sonuç'].values[0]

    def reset(self):
        self.d1p = np.random.randint(1, 101) 
        self.d2p = np.random.randint(1, 101) 
        self.current_step = 0
        return self._get_state()

    def step(self, action_idx):
        self.current_step += 1

        # Seçilen aksiyonları al
        action1, action2 = self.action_space[action_idx]
        self.previous_location = self.new_location
        
        # d1 yönünde hareket
        if action1 == "forward" and self.d1p < 100:
            self.d1p += 1
        elif action1 == "backward" and self.d1p > 1:
            self.d1p -= 1
        elif action1 == "hold":
            pass

        # d2 yönünde hareket
        if action2 == "forward" and self.d2p < 100:
            self.d2p += 1
        elif action2 == "backward" and self.d2p > 1:
            self.d2p -= 1
        elif action2 == "hold":
            pass 

        

        new_location = self._get_location(self.d1p, self.d2p)
        self.new_location = new_location
        self.highest_avt = max(self.highest_avt, self.new_location)
        
        reward = 0
        threshold = 40
        if new_location > threshold and new_location > self.previous_location: 
            reward = (new_location - 40)
            """
        elif 48 < new_location < threshold: 
            reward = (new_location - 40) ** 0.7
            """
        elif new_location > self.previous_location:
            reward = 1
        elif new_location < self.previous_location: 
            reward = -1  

      
        random_factor = 0.1  
        reward += random_factor * (2 * np.random.random() - 1) 

        
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done




    def _get_state(self):
        # Mevcut konumu döndür
        state = np.array([self.d1p, self.d2p], dtype=np.float32)
        return state

    def render(self):
        print(f"Current position: (d1={self.d1p}, d2={self.d2p}), Reflectance: {self.result[(self.result['d1'] == self.d1p) & (self.result['d2'] == self.d2p)]['sonuç'].values[0]}")
