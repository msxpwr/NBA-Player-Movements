# %%
from Game import Game
from time import time
import pandas as pd
game = Game(game_id='0021500492')
df, action_df = game.read_json()
moment_df, game_series, move_df, player_df, team_df, action_df = game.get_dfs()

# %%






















    