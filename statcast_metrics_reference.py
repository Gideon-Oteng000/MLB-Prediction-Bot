#!/usr/bin/env python3
"""
Statcast Metrics Reference Tool
Comprehensive listing of all pitcher and hitter Statcast metrics available in pybaseball
"""

import pybaseball
from datetime import datetime, timedelta
import pandas as pd


def display_header(title):
    """Display a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def display_section(title):
    """Display a formatted section header"""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")


def get_sample_statcast_data():
    """Get a small sample of Statcast data to examine available columns"""
    try:
        # Get last 3 days of data (small sample)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        print("Fetching sample Statcast data to analyze available metrics...")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Get Statcast data
        data = pybaseball.statcast(
            start_dt=start_date.strftime('%Y-%m-%d'),
            end_dt=end_date.strftime('%Y-%m-%d')
        )

        return data
    except Exception as e:
        print(f"Error fetching Statcast data: {e}")
        return None


def list_pitcher_metrics():
    """List all pitcher-related Statcast metrics"""

    display_section("PITCHER STATCAST METRICS")

    pitcher_metrics = {
        "Velocity Metrics": [
            "release_speed",
            "effective_speed",
            "release_speed_deprecated"
        ],

        "Movement Metrics": [
            "pfx_x",  # Horizontal movement
            "pfx_z",  # Vertical movement
            "plate_x",  # Horizontal location at plate
            "plate_z",  # Vertical location at plate
            "sz_top",   # Top of strike zone
            "sz_bot",   # Bottom of strike zone
        ],

        "Release Point Metrics": [
            "release_pos_x",  # Horizontal release position
            "release_pos_y",  # Distance from home plate at release
            "release_pos_z",  # Height at release
            "release_extension",  # Extension toward home plate
        ],

        "Spin Metrics": [
            "release_spin_rate",  # Spin rate (RPM)
            "spin_axis",          # Spin axis (degrees)
        ],

        "Break Metrics": [
            "break_angle_deprecated",
            "break_length_deprecated",
            "break_y_deprecated",
        ],

        "Result Metrics": [
            "launch_speed",       # Exit velocity when ball is hit
            "launch_angle",       # Launch angle when ball is hit
            "estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle",
            "woba_value",
            "woba_denom",
            "babip_value",
            "iso_value",
        ],

        "Expected Metrics": [
            "hit_distance_sc",    # Hit distance (Statcast)
            "post_away_score",
            "post_home_score",
            "post_bat_score",
            "post_fld_score",
        ]
    }

    for category, metrics in pitcher_metrics.items():
        print(f"\n{category}:")
        for metric in metrics:
            print(f"  • {metric}")

    return pitcher_metrics


def list_hitter_metrics():
    """List all hitter-related Statcast metrics"""

    display_section("HITTER STATCAST METRICS")

    hitter_metrics = {
        "Contact Quality": [
            "launch_speed",       # Exit velocity
            "launch_angle",       # Launch angle
            "hit_distance_sc",    # Projected hit distance
            "barrel",             # Barrel classification (1 if barrel, 0 if not)
        ],

        "Expected Performance": [
            "estimated_ba_using_speedangle",  # Expected batting average
            "estimated_woba_using_speedangle",  # Expected wOBA
            "woba_value",         # Actual wOBA value for the event
            "woba_denom",         # Whether event counts in wOBA denominator
            "babip_value",        # BABIP value for the event
            "iso_value",          # ISO value for the event
        ],

        "Plate Location": [
            "plate_x",            # Horizontal location of pitch
            "plate_z",            # Vertical location of pitch
            "sz_top",             # Top of batter's strike zone
            "sz_bot",             # Bottom of batter's strike zone
        ],

        "Swing Metrics": [
            "swing_length",       # Length of swing path
            "bat_speed",          # Bat speed at contact
            "swing_length_sc",    # Swing length (Statcast)
        ],

        "Direction/Spray": [
            "hit_location",       # Fielding position number
            "hc_x",              # Hit coordinate X (horizontal)
            "hc_y",              # Hit coordinate Y (vertical)
            "spray_angle",        # Spray angle of batted ball
        ],

        "Situational": [
            "on_3b",             # Runner on 3rd base
            "on_2b",             # Runner on 2nd base
            "on_1b",             # Runner on 1st base
            "outs_when_up",      # Number of outs
            "inning",            # Inning number
            "inning_topbot",     # Top or bottom of inning
            "balls",             # Ball count
            "strikes",           # Strike count
        ]
    }

    for category, metrics in hitter_metrics.items():
        print(f"\n{category}:")
        for metric in metrics:
            print(f"  • {metric}")

    return hitter_metrics


def list_general_statcast_metrics():
    """List general Statcast metrics that apply to both pitchers and hitters"""

    display_section("GENERAL STATCAST METRICS")

    general_metrics = {
        "Game Information": [
            "game_date",
            "game_pk",
            "player_name",
            "batter",
            "pitcher",
            "home_team",
            "away_team",
        ],

        "Pitch Classification": [
            "pitch_type",         # Type of pitch (FF, SL, CH, etc.)
            "pitch_name",         # Full pitch name
            "description",        # Outcome description
            "zone",              # Strike zone location (1-14)
            "type",              # Ball, Strike, or In Play
        ],

        "Outcome Classification": [
            "events",            # Result of at-bat
            "bb_type",           # Batted ball type (ground_ball, line_drive, etc.)
            "if_fielding_alignment",
            "of_fielding_alignment",
        ],

        "Advanced Classifications": [
            "hit_location",       # Where ball was fielded
            "fielder_2",          # Catcher involvement
            "fielder_3",          # First baseman involvement
            "fielder_4",          # Second baseman involvement
            "fielder_5",          # Third baseman involvement
            "fielder_6",          # Shortstop involvement
            "fielder_7",          # Left fielder involvement
            "fielder_8",          # Center fielder involvement
            "fielder_9",          # Right fielder involvement
        ]
    }

    for category, metrics in general_metrics.items():
        print(f"\n{category}:")
        for metric in metrics:
            print(f"  • {metric}")

    return general_metrics


def get_leaderboard_metrics():
    """List metrics available in pybaseball leaderboards"""

    display_section("PYBASEBALL LEADERBOARD METRICS")

    print("\nPitcher Leaderboard Metrics (pitching_stats):")
    pitcher_leaderboard = [
        "W", "L", "ERA", "G", "GS", "CG", "SHO", "SV", "BS", "IP", "H", "R", "ER",
        "HR", "BB", "IBB", "HBP", "WP", "BK", "SO", "WHIP", "BABIP", "LOB%", "FIP",
        "GB/FB", "LD%", "GB%", "FB%", "IFFB%", "HR/FB", "IFH%", "BUH%", "Pull%",
        "Cent%", "Oppo%", "Soft%", "Med%", "Hard%", "Swing%", "Strike%", "Contact%",
        "Zone%", "F-Strike%", "SwStr%", "ERA-", "FIP-", "xFIP", "WAR"
    ]

    for i, metric in enumerate(pitcher_leaderboard):
        if i % 8 == 0:
            print()
        print(f"{metric:>8}", end=" ")

    print("\n\nHitter Leaderboard Metrics (batting_stats):")
    hitter_leaderboard = [
        "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "IBB", "SO", "HBP",
        "SF", "SH", "GDP", "SB", "CS", "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
        "GB/FB", "LD%", "GB%", "FB%", "IFFB%", "HR/FB", "IFH%", "BUH%", "Pull%",
        "Cent%", "Oppo%", "Soft%", "Med%", "Hard%", "Swing%", "Strike%", "Contact%",
        "Zone%", "F-Strike%", "SwStr%", "wOBA", "wRAA", "wRC+", "BSR", "Off", "Def", "WAR"
    ]

    for i, metric in enumerate(hitter_leaderboard):
        if i % 8 == 0:
            print()
        print(f"{metric:>8}", end=" ")

    print("\n")


def list_available_functions():
    """List all available pybaseball functions for getting Statcast data"""

    display_section("PYBASEBALL STATCAST FUNCTIONS")

    functions = {
        "Raw Statcast Data": [
            "statcast(start_dt, end_dt)",
            "statcast_single_game(game_pk)",
            "statcast_pitcher(start_dt, end_dt, player_id)",
            "statcast_batter(start_dt, end_dt, player_id)",
        ],

        "Leaderboards": [
            "pitching_stats(start_season, end_season, league='all', qual=1)",
            "batting_stats(start_season, end_season, league='all', qual=1)",
            "pitching_stats_bref(season, league='all')",
            "batting_stats_bref(season, league='all')",
        ],

        "Player Lookups": [
            "playerid_lookup(last, first)",
            "playerid_reverse_lookup(player_id, key_type='mlbam')",
            "chadwick_register()",
        ],

        "Team Data": [
            "team_pitching(start_season, end_season, team=None)",
            "team_batting(start_season, end_season, team=None)",
            "schedule_and_record(season, team)",
        ],

        "Historical Data": [
            "lahman_pitching()",
            "lahman_batting()",
            "retrosheet_data(season)",
        ]
    }

    for category, funcs in functions.items():
        print(f"\n{category}:")
        for func in funcs:
            print(f"  • {func}")


def demonstrate_usage():
    """Show example usage of the metrics"""

    display_section("EXAMPLE USAGE")

    example_code = '''
# Example 1: Get Statcast data for a date range
from datetime import datetime, timedelta
import pybaseball as pyb

# Get last week's data
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
data = pyb.statcast(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# Example 2: Analyze pitcher metrics
pitcher_data = data[['pitcher', 'player_name', 'release_speed', 'release_spin_rate',
                     'pfx_x', 'pfx_z', 'launch_speed', 'launch_angle']].dropna()

# Example 3: Analyze hitter metrics
hitter_data = data[['batter', 'player_name', 'launch_speed', 'launch_angle',
                    'hit_distance_sc', 'estimated_ba_using_speedangle', 'barrel']].dropna()

# Example 4: Get season leaderboards
pitching_leaders = pyb.pitching_stats(2024, 2024, qual=50)
batting_leaders = pyb.batting_stats(2024, 2024, qual=100)

# Example 5: Player lookup
player_info = pyb.playerid_lookup('Trout', 'Mike')
player_id = player_info['key_mlbam'].iloc[0]

# Example 6: Player-specific Statcast data
trout_data = pyb.statcast_batter('2024-04-01', '2024-09-30', player_id)
'''

    print(example_code)


def main():
    """Main function to display all Statcast metrics"""

    display_header("PYBASEBALL STATCAST METRICS REFERENCE GUIDE")

    print("\nThis script provides a comprehensive reference for all Statcast metrics")
    print("available through the pybaseball library for both pitchers and hitters.")
    print("\nNote: This is a reference guide. Actual data availability may vary")
    print("depending on the date range and specific games queried.")

    # List all the metrics
    pitcher_metrics = list_pitcher_metrics()
    hitter_metrics = list_hitter_metrics()
    general_metrics = list_general_statcast_metrics()

    # Show leaderboard metrics
    get_leaderboard_metrics()

    # Show available functions
    list_available_functions()

    # Show usage examples
    demonstrate_usage()

    # Summary
    display_section("SUMMARY")

    total_pitcher = sum(len(metrics) for metrics in pitcher_metrics.values())
    total_hitter = sum(len(metrics) for metrics in hitter_metrics.values())
    total_general = sum(len(metrics) for metrics in general_metrics.values())

    print(f"""
Total Metrics Available:
• Pitcher-specific metrics: {total_pitcher}
• Hitter-specific metrics: {total_hitter}
• General/game metrics: {total_general}
• Total unique metrics: {total_pitcher + total_hitter + total_general}

Key Resources:
• PyBaseball Documentation: https://github.com/jldbc/pybaseball
• Statcast Glossary: https://www.mlb.com/glossary/statcast
• Baseball Savant: https://baseballsavant.mlb.com/

Installation:
pip install pybaseball

Basic Usage:
import pybaseball as pyb
data = pyb.statcast('2024-04-01', '2024-04-07')
""")


if __name__ == "__main__":
    main()