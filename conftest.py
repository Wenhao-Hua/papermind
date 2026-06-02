"""Put the repo root on sys.path so top-level research packages (trainer/,
evaluation/) import cleanly under pytest, alongside the papermind package."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
