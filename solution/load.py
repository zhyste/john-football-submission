import os
import sys
import time
import numpy as np 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games import DATA_DIR, GAME, GAMES, require, source_frames 
from starter.baseline import team_of_each_track

# Parsed arrays are cached under data/ so they stay gitignored with the rest of the match data
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tracking.npz")

def readTracking(rawPath,nFrames,nSlots,pitch):
    length, width = pitch 

    #Created 6 empty tables, filled with nan 
    #players not visible stays as nan 
    players_x = np.full((nFrames,nSlots),np.nan,dtype=np.float32)
    players_y = np.full((nFrames,nSlots),np.nan,dtype=np.float32)
    players_s = np.full((nFrames,nSlots),np.nan,dtype=np.float32)
    ball_x = np.full(nFrames,np.nan,dtype=np.float32)
    ball_y = np.full(nFrames,np.nan,dtype=np.float32)
    ball_s = np.full(nFrames,np.nan,dtype=np.float32)
    
    nLines = 0 
    
    with open(rawPath) as handle:
        for line in handle:
            parts = line.strip().split(':') #cutting the line data splitting by : 
            if len(parts) != 3:
                continue 
            
            framePart, playersPart, ballPart = parts 
            
            
            frame = int(framePart) - 1
            
            numbers = playersPart.replace(';',',').split(',')
            
            players = np.array(numbers,dtype=np.float64).reshape(nSlots,3)
            #player data gives a number from 0-1, speed is already in m/s
            players_x[frame] = players[:,0] * length 
            players_y[frame] = players[:,1] * width  
            players_s[frame] = players[:,2]
            
            ball = ballPart.split(',')
            ball_x[frame] = float(ball[0])*length
            ball_y[frame] = float(ball[1])*width
            ball_s[frame] = float(ball[2])
            
            nLines += 1
            
    return players_x,players_y,players_s,ball_x,ball_y,ball_s,nLines

def loadTracking(refresh=False, fix_goalkeepers=None):
    """A dict of the six tables plus team_of_slot, reading the raw file only once.

    fix_goalkeepers (default: atd_fix.ENABLED): the goalkeeper slots get the team
    that defends the half they are in, and the tables have 48 slots (atd_fix.py). False gives
    the raw 46-slot ATD tracking.
    """
    if fix_goalkeepers is None:
        import atd_fix
        fix_goalkeepers = atd_fix.ENABLED
    tracking = _load_raw_tracking(refresh)
    if fix_goalkeepers:
        import atd_fix
        tracking = atd_fix.corrected_tracking(tracking)
    return tracking


def _load_raw_tracking(refresh=False):
    if os.path.exists(CACHE_PATH) and not refresh:
        with np.load(CACHE_PATH) as cached:
            return dict(cached)

    metaPath, rawPath, = require(['atd_meta','atd_raw'])
    team_of_slot = team_of_each_track(metaPath)
    n_frames = source_frames("tactical")
    tables = readTracking(rawPath, n_frames, len(team_of_slot), GAMES[GAME]["pitch"])
    names = ["players_x","players_y","players_s","ball_x","ball_y","ball_s"]
    tracking = dict(zip(names, tables[:6]))
    tracking["team_of_slot"] = team_of_slot

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_PATH, **tracking)
    return tracking

if __name__ == '__main__':
    start = time.time()
    tracking = loadTracking(refresh=True, fix_goalkeepers=False)
    team_of_slot = tracking["team_of_slot"]
    ball_x = tracking["ball_x"]
    n_frames = source_frames("tactical")
    n_lines = int(np.any(~np.isnan(tracking["players_x"]), axis=1).sum())
    
    # Sanity checks. Only counts and percentages are printed
    print("frames with players:", n_lines, "of", n_frames)
    print("track slots:     ", len(team_of_slot),
    "| home", int((team_of_slot == 0).sum()), "| away", int((team_of_slot == 1).sum()))
    # np.isnan(ball_x) is True where the ball is missing;flips it to "ball seen".
    # np.mean of True/False values = the share of frames where it's True.
    print("ball seen:       ", round(100 * np.mean(~np.isnan(ball_x)), 1), "% of frames")
    print("seconds taken:   ", round(time.time() - start, 1))
    print("cached to:       ", CACHE_PATH)
