#!/usr/bin/env python3

print("Testing imports...")

try:
    import torch
    print("✓ torch imported successfully")
except ImportError as e:
    print(f"✗ torch import failed: {e}")

try:
    import numpy as np
    print(f"✓ numpy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import wandb
    print("✓ wandb imported successfully")
except ImportError as e:
    print(f"✗ wandb import failed: {e}")

try:
    from transformers import AutoTokenizer
    print("✓ transformers.AutoTokenizer imported successfully")
except ImportError as e:
    print(f"✗ transformers.AutoTokenizer import failed: {e}")

try:
    from peft import LoraConfig
    print("✓ peft.LoraConfig imported successfully")
except ImportError as e:
    print(f"✗ peft.LoraConfig import failed: {e}")

try:
    from trl import AutoModelForCausalLMWithValueHead
    print("✓ trl.AutoModelForCausalLMWithValueHead imported successfully")
except ImportError as e:
    print(f"✗ trl.AutoModelForCausalLMWithValueHead import failed: {e}")

try:
    import gymnasium as gym
    print("✓ gymnasium imported successfully")
except ImportError as e:
    print(f"✗ gymnasium import failed: {e}")

try:
    from llamagym import Agent
    print("✓ llamagym.Agent imported successfully")
except ImportError as e:
    print(f"✗ llamagym.Agent import failed: {e}")

print("Import testing complete!") 