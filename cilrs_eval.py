import os
import torch
import yaml

from carla_env.env import Env


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self):
        model = CILRS()
        model.load_state_dict(torch.load(self.config["model_path"], map_location=torch.device('cpu')))
        model.eval()
        return model
        

    def generate_action(self, rgb, command, speed):
        with torch.no_grad():
            img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().div(255.0)
            speed = torch.tensor(speed).unsqueeze(0).float()
            command = torch.tensor(command).unsqueeze(0).float()
            action = self.agent(img, command)
            throttle = action[0][0].item()
            brake = action[0][1].item()
            steer = action[0][2].item()
        return throttle, steer, brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(self, rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(self,state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
