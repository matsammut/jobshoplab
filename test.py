from pathlib import Path

import requests

from jobshoplab import JobShopLabEnv, load_config
from jobshoplab.env.factories.rewards import RewardFactory

OLLAMA_URL = "https://llm.mushy.fr"

def query_ollama(prompt: str, model: str = "llama2") -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]

def should_penalize(obs, info, reward, threshold=-10):
    prompt = f"""
You are an expert in job shop scheduling.

Based on the following environment state:
- Observation: {obs}
- Info: {info}

My algorithm gave the following reward ()
- Reward: {reward}

Evaluate
"""
    response = query_ollama(prompt)
    return response

# Load a pre-defined configuration
config = load_config(config_path=Path("data/config/getting_started_config.yaml"))

# Create the environment
env = JobShopLabEnv(config=config)

# Run with random actions until done
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, truncated, terminated, info = env.step(action)

    # Query the LLM about whether to punish the action
    print(reward)
    # llm_judgment = should_penalize(obs, info, reward)

    # print(f"LLM says: {llm_judgment}")

    # if "yes" in llm_judgment.lower():
    #     reward -= 100

    done = truncated or terminated
# Visualize the final schedule
env.render()

