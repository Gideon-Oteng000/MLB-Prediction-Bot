#!/usr/bin/env python3
"""
Quick test of the enhanced LineupManager
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlb_hr_clean_v4 import LineupManager

def test_lineup_manager():
    """Test the LineupManager to verify it fetches games from multiple sources"""
    print("Testing Enhanced LineupManager...")
    print("=" * 50)

    try:
        manager = LineupManager()
        games = manager.get_all_todays_games()

        if games:
            print(f"\n✅ SUCCESS: Retrieved {len(games)} games")

            # Count by source
            source_counts = {}
            for game in games:
                source = game.get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1

            print("\nBreakdown by source:")
            for source, count in source_counts.items():
                print(f"   {source}: {count} games")

            # Show first few games
            print(f"\nFirst 3 games:")
            for i, game in enumerate(games[:3], 1):
                print(f"   {i}. {game['away_team']} @ {game['home_team']} ({game['source']})")
                print(f"      Home lineup: {len(game['home_lineup'])} players")
                print(f"      Away lineup: {len(game['away_lineup'])} players")
        else:
            print("❌ No games retrieved")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_lineup_manager()