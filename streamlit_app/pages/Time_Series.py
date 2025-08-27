import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Base project directory (nigeria_food_prices/)
BASE_DIR = Path(__file__).parent.parent

# Add src directory (for your reusable modules)
sys.path.append(str(BASE_DIR / "src"))
