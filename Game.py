import pandas as pd
from Event import Event
from Constant import Constant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from matplotlib.patches import Circle, Rectangle, Arc

class Game:
    """A class for keeping info about the games"""
        
    def __init__(self, game_id):
        # self.events = None
        self.game_id = game_id
    
    def remove_duplicated_moments(self, sportvu_data, action_df):
        pd.set_option('mode.chained_assignment', None)
        
        sportvu_events_data = sportvu_data['events']
        last_moments = None
        mask = [True for x in range(len(sportvu_events_data))]
        saved_events = []
        periods = []
        
        for idx, event_data in enumerate(sportvu_events_data):
            # naive assumption that one event always in same period
            # skip if no moments (often rebound or make)
            # TODO: check what about event_type when skip
            try:
                period = event_data['moments'][0][0]
                milisecond = event_data['moments'][0][1]
                if 1451698355021== milisecond:
                    a = 1+1
            except:
                mask[idx] = False
                last_moments = new_moments
                continue
                 
            # mask events with same moments (when few eventmsgtype in one event)
            new_moments = event_data['moments']
            if last_moments == new_moments:
                mask[idx] = False
                same_event = True
            else:
                last_moments = new_moments
                same_event = False
                
            """event_id = int(event_data['eventId'])
            event_type_data = action_df[action_df['EVENTNUM'] == event_id]
            if event_type_data.empty:
                if same_event:
                    mask[idx] = False
                else:
                    saved_events.append([])
            else:
                if same_event:
                    saved_events[-1].append(event_type_data.iloc[0])
                else:
                    saved_events.append([event_type_data.iloc[0]])"""
            
        sportvu_data = sportvu_data[mask]
        sportvu_events_data = sportvu_data['events']
        #sportvu_data['actions'] = saved_events
        
        pd.set_option('mode.chained_assignment', 'warn')
        return sportvu_data, sportvu_events_data
    
    def find_players_exceeding_threshold(self, data, threshold):
        differences = np.diff(data, axis=0)
        distances = np.linalg.norm(differences, axis=2)
        # więcej niż 3 zawodników się teleportowało
        exceeding_threshold_indices = np.sum(distances > threshold, axis=1) > 3
        return exceeding_threshold_indices


    def split_event(row, glitches):
        pass
    
    def split_glitched_events(self, df):
        MAX_PLAYER_MOVE = 1
        good_events = []
        for index, row in df.iterrows():
            new_moments = row['events']['moments']
            #period = row['period']
            moves = np.array(np.array([x[5] for x in new_moments]))
            player_moves = moves[:,1:,1:3]
            glitches = self.find_players_exceeding_threshold(player_moves, MAX_PLAYER_MOVE)
            new_events = split_event(row, glitches)
            good_events += new_events
                # znajdz nagle roznice odleglosci i podziel
                # co zrobic z krotkim
                # co zrobic gdy jest NaN
                # znajdz wolne i co
                # rysowanie
                # heatmapa ruchu
                # heatmapa ze wzgledu na typ
                # rzuty prawdopodobienstwo
                # rzuty a czas do konca rzutu
            
            
            
        return df
    
    def pctimestring_to_quater_time(self, pctimestring):
        minutes, seconds = pctimestring.split(':')
        minutes, seconds = int(minutes), int(seconds)
        seconds = minutes * 60 + seconds
        return seconds

    def quater_time_to_play_time(self, quater_time, quater):
        play_time = (int(quater) - 1) * 720 + (720 - quater_time)
        return round(play_time, 2)
    
    
    def get_game_series(self, sportvu_data, sportvu_events_data):
        game_series = sportvu_data[['gameid', 'gamedate']].iloc[0]
        game_series = game_series.rename({'gameid': 'game_id'})
        game_series['team_home_id'] = sportvu_events_data.iloc[0]['home']['teamid']
        game_series['team_visitor_id'] = sportvu_events_data.iloc[0]['visitor']['teamid']
        return game_series
    
    def get_team_df(self, sportvu_events_data):
        teams = [sportvu_events_data.iloc[0]['home'], 
                 sportvu_events_data.iloc[0]['visitor']]
        team_df = pd.DataFrame([{
            'team_id' : team['teamid'],
            'team_name': team['name'],
            'abbreviation': team['abbreviation']
            } for team in teams])
        team_df['color'] = team_df['team_id'].map(Constant.TEAMS_COLOR)
        return team_df
    
    def get_player_df(self, sportvu_events_data):
        teams = [sportvu_events_data.iloc[0]['home'], 
                 sportvu_events_data.iloc[0]['visitor']]


        player_df = pd.DataFrame([
            {
                'player_id': player['playerid'],
                'team_id': team['teamid'],
                'lastname': player['lastname'],
                'firstname': player['firstname'],
                'jersey_num': player['jersey'],
                'position': player['position']
            }
            for team in teams for player in team['players']
        ])
        return player_df
    
    def get_moment_df(self, sportvu_events_data, game_id):
        sporvu_events_df = pd.DataFrame(sportvu_events_data.to_list())
        moment_df = pd.DataFrame([
            {
                'game_id': game_id,
                'quater': moment[0],
                'time_usa': moment[1],
                'quater_time': moment[2],
                'shot_clock': moment[3],
                'moves': moment[5]
            }
            for events_moments in sporvu_events_df['moments'] 
            for moment in events_moments
        ])
        moment_df = moment_df[moment_df.duplicated(subset=['time_usa', 'quater_time']) == False]
        moment_df['play_time'] = moment_df[['quater_time', 'quater']].apply(lambda row: self.quater_time_to_play_time(*row), axis=1)
        moment_df.sort_values(by=['quater', 'play_time'], inplace=True)
        moment_df = moment_df[moment_df['moves'].apply(lambda x: len(x) == 11)]
        moment_df.insert(0, 'moment_id', range(len(moment_df)))
        return moment_df
    
    def get_move_df(self, moment_df):
        move_df = pd.DataFrame([
            {
                'moment_id': moment.moment_id,
                'x': move[2],
                'y': move[3],
                'z': move[4],
                'object_id': move[1],
            }
            for moment in  moment_df.itertuples() 
            for move in moment.moves
        ])
        move_df.insert(0, 'move_id', range(len(move_df)))
        return move_df
    
    def get_action_df(self, action_df, moment_df):
        action_df['quater_time'] = action_df['PCTIMESTRING'].apply(self.pctimestring_to_quater_time)
        action_df['play_time'] = action_df[['quater_time', 'PERIOD']].apply(lambda row: self.quater_time_to_play_time(*row), axis=1)
        #action_df['play_time'] = action_df['play_time'].apply(lambda x: x + 0.02)

        tmp_df = moment_df[['play_time', 'moment_id']]
        action_df['play_time'] = action_df['play_time'].astype(float)
        action_df = pd.merge_asof(action_df, tmp_df, on='play_time', direction='nearest')
        return action_df
    
    def get_side(self, move_df, moment_id):
        """
        Return True if ball left side
        """
        try:
            x = move_df[move_df['moment_id'] == moment_id]['x'].iloc[0]
            return x < 50
        except IndexError:
            return None
        
    def get_home_side(self, shot_df, move_df, quater):
        # True = home is Right/+, False = home is Left/-
        SHOT_REQUIRED_NUM = 5
        quater_shot_df = shot_df[shot_df['PERIOD']==quater]
        
        quater_shot_counter = 0
        proof_counter = 0
        for row in quater_shot_df.itertuples():
            
            side = self.get_side(move_df, row.moment_id)
            if side is np.nan:
                continue
            
            if row.HOMEDESCRIPTION is not np.nan and ('MISS' in row.HOMEDESCRIPTION or
                                                      'PTS' in row.HOMEDESCRIPTION):
                if side:
                    proof_counter += 1
                else:
                    proof_counter -= 1
            else:
                if side:
                    proof_counter -= 1
                else:
                    proof_counter += 1
            
            if quater_shot_counter == SHOT_REQUIRED_NUM:
                break
            quater_shot_counter += 1
            
            return proof_counter > 0
        
    def reverse_coordinates(self, df):
        x_symetric = 50
        y_symetric = 25
        df['x'] = 2 * x_symetric - df['x']
        df['y'] = 2 * y_symetric - df['y']
        return df
    
    def home_side_to_right(self, moment_df, shot_df, move_df):
        for quater in set(moment_df['quater']): # should be .unique
            is_home_right_side = self.get_home_side(shot_df, move_df, quater) 
            if is_home_right_side:
                pass
            else:
                print(f'swap side quater {quater}')
                quater_moment_ids = moment_df[moment_df['quater'] == quater]['moment_id']
                condition = move_df['moment_id'].isin(quater_moment_ids)
        
                move_df[condition] = move_df[condition].apply(self.reverse_coordinates, 
                                                            axis=1)
        return move_df
    
    def read_json(self):
        sportvu_data = pd.read_json(f'./data/sportvu/{self.game_id}.json')
        action_df = pd.read_csv(f'./data/events/{self.game_id}.csv')

        sportvu_data, sportvu_events_data = self.remove_duplicated_moments(sportvu_data, 
                                                                           action_df)
        
        
        game_series = self.get_game_series(sportvu_data, sportvu_events_data)        
        team_df = self.get_team_df(sportvu_events_data)
        player_df = self.get_player_df(sportvu_events_data)
        moment_df = self.get_moment_df(sportvu_events_data, game_series.game_id)
        move_df = self.get_move_df(moment_df)
        moment_df.drop('moves', axis=1)
        action_df = self.get_action_df(action_df, moment_df)
        shot_df = action_df[action_df['EVENTMSGTYPE'].isin([1, 2, 3])]
        move_df = self.home_side_to_right(moment_df, shot_df, move_df)
        
        self.moment_df = moment_df
        self.game_series = game_series
        self.move_df = move_df
        self.player_df = player_df
        self.team_df = team_df
        self.action_df = action_df
        return sportvu_data, sportvu_events_data    
        
    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info, moment_df, players_df, ball_df, table_cells):
        #print(f'frame {i}')
        actual_moment = moment_df.iloc[i]
        moment_id = actual_moment['moment_id']
        actual_players = players_df[players_df['moment_id'] == moment_id]
        actual_ball = ball_df[ball_df['moment_id'] == moment_id].iloc[0]
        
        for idx, cell in enumerate(table_cells[:10]):
            player = actual_players.iloc[idx]
            cell._text.set_color('white')
            cell._text.set_text(f'{player.lastname} {player.firstname} #{player.jersey_num}')
            
        for j, circle in enumerate(player_circles):
            player = actual_players.iloc[j]
            circle.center = player.x, player.y
            annotations[j].set_position(circle.center)
            annotations[j].set_text(str(player.jersey_num))
            try:
                clock_test = 'Quarter {:d}\n Play_time {:03.1f}\n {:02d}:{:02d}\n {:03.1f}'.format(
                            int(actual_moment.quater),
                            actual_moment.play_time,
                            int(actual_moment.quater_time) % 3600 // 60,
                            int(actual_moment.quater_time) % 60,
                            actual_moment.shot_clock)
            except Exception:
                clock_test = 'Quarter {:d}\n Play_time {:03.1f}\n {:02d}:{:02d}\n NaN'.format(
                            int(actual_moment.quater),
                            actual_moment.play_time,
                            int(actual_moment.quater_time) % 3600 // 60,
                            int(actual_moment.quater_time) % 60)
                
            clock_info.set_text(clock_test)
        ball_circle.center = actual_ball.x, actual_ball.y
        ball_circle.radius = actual_ball.z / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle
    
    
    def show(self, start_time, end_time, file_name='tmp.gif'):
        show_moment_df = self.moment_df[(self.moment_df['play_time'] >= start_time) & (self.moment_df['play_time'] <= end_time)]
        show_moment_df = show_moment_df[['moment_id', 'quater', 'quater_time', 'play_time', 'shot_clock']]

        show_move_df = self.move_df[self.move_df['moment_id'].isin(show_moment_df['moment_id'])]    

        players_df = show_move_df[show_move_df['object_id'] != -1]
        players_df = pd.merge(players_df, self.player_df, how='left', left_on='object_id', right_on='player_id')
        players_df = pd.merge(players_df, self.team_df, how='left', on='team_id')
        players_df = players_df[['x', 'y', 'jersey_num', 'lastname', 'firstname', 'team_id', 'color', 'moment_id', 'abbreviation']]

        ball_df = show_move_df[show_move_df['object_id'] == -1]
        ball_df['color'] = Constant.BALL_COLOR

        home_player_df = players_df[players_df['team_id'] == self.game_series['team_home_id']] 
        home_team_data = home_player_df[['abbreviation' , 'color']].iloc[0]
        visitor_player_df = players_df[players_df['team_id'] == self.game_series['team_visitor_id']]
        visitor_team_data = visitor_player_df[['abbreviation' , 'color']].iloc[0]
        
        start_moment = show_moment_df.iloc[0]
        moment_id = start_moment['moment_id']
        actual_players = players_df[players_df['moment_id'] == moment_id]
        actual_ball = ball_df[ball_df['moment_id'] == moment_id].iloc[0]
        
        ax = plt.axes(xlim=(Constant.X_MIN,
                    Constant.X_MAX),
              ylim=(Constant.Y_MIN,
                    Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)
        
        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                         color='black', horizontalalignment='center',
                           verticalalignment='center')

        annotations = [ax.annotate(player.jersey_num, xy=[0, 0], color='w',
                                horizontalalignment='center',
                                verticalalignment='center', fontweight='bold')
                    for player in actual_players.itertuples()]
        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                        for player in actual_players.itertuples()]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                color=actual_ball.color)
        """ax.annotate('.', xy=[20, 20], color='b',
                                horizontalalignment='center',
                                verticalalignment='center', fontweight='bold')
        ax.annotate('.', xy=[22, 20], color='b',
                        horizontalalignment='center',
                        verticalalignment='center', fontweight='bold')"""
        
        column_labels = tuple([visitor_team_data.abbreviation, home_team_data.abbreviation])
        column_colours = tuple([visitor_team_data.color, home_team_data.color])
        cell_colours = [column_colours for _ in range(5)]
        table = plt.table(cellText=[('', '') for x in range(5)],
                            colLabels=column_labels,
                            colColours=column_colours,
                            colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                            loc='bottom',
                            cellColours=cell_colours,
                            fontsize=Constant.FONTSIZE,
                            cellLoc='center')
        #table.scale(1, Constant.SCALE)
        table_cells = table.properties()['children']
        for cell in table_cells:
            cell._text.set_color('white')


        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)
        
        court = plt.imread("court.png")
        anim = animation.FuncAnimation(
            fig, self.update_radius,
            fargs=(player_circles, ball_circle, annotations, clock_info, show_moment_df, players_df, ball_df, table_cells),
            frames=len(show_moment_df), interval=15)


        ax.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN])

        #plt.show()
        anim.save(file_name, dpi=200)
        
    def count_dist(self, ball, player):
        return math.sqrt(pow(player.x - ball.x, 2) + pow(player.y - ball.y, 2))
        
    def fix_shot_moment(self, shot, ball_df):
        MAX_DIST_BALL_SHOOTER = 2
        UNACCURATE_THRESHOLD = 25 * 24 # 25hz * 24s shot clock
        
        moment_id = shot.moment_id
        shooter_id = shot.PLAYER1_ID
        shooter_move_df = self.move_df[self.move_df['object_id'] == shooter_id]
        
        actual_ball = ball_df[ball_df['moment_id'] == moment_id].iloc[0]
        actual_shooter = shooter_move_df[shooter_move_df['moment_id'] == moment_id].iloc[0]
        distance = self.count_dist(actual_ball, actual_shooter)
        counter = 0
        while distance > MAX_DIST_BALL_SHOOTER:
            moment_id -= 1
            
            actual_ball = ball_df[ball_df['moment_id'] == moment_id].iloc[0]
            actual_shooter = shooter_move_df[shooter_move_df['moment_id'] == moment_id].iloc[0]
            distance = self.count_dist(actual_ball, actual_shooter)
            if counter > UNACCURATE_THRESHOLD: 
                print('Unaccurate repair shot num {shot.EVENTNUM}')
            counter += 1
        shot.moment_id = moment_id
        return shot

    def get_shot_df(self):
        shot_df = self.action_df[self.action_df['EVENTMSGTYPE'].isin([1,2,3])]
        ball_df = self.move_df[self.move_df['object_id'] == -1]
        shot_df = shot_df.apply(lambda shot: self.fix_shot_moment(shot, ball_df), axis=1)
        shot_df = shot_df.drop(['play_time', 'quater_time'], axis=1)
        merge_df = self.moment_df[['moment_id', 'quater_time', 'play_time']]
        shot_df = pd.merge(shot_df, merge_df, how='left', on='moment_id')
        return shot_df
    
    def get_dfs(self):
        return self.moment_df, self.game_series, self.move_df, self.player_df, self.team_df, self.action_df

    