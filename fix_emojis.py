"""
Fix emoji encoding issues by replacing emojis with text
"""

import re

def fix_emojis(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace emojis with text equivalents
    replacements = {
        '📋': '[INFO]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '✅': '[SUCCESS]',
        '📅': '[SCHEDULE]',
        '⏰': '[TIME]',
        '📊': '[DATA]',
        '🎮': '[GAME]',
        '📈': '[STATS]',
        '🔍': '[DEBUG]',
        '🏠': '[HOME]',
        '✈️': '[AWAY]',
        '🎉': '[SUCCESS]',
        '💡': '[TIP]',
        '🎯': '[TARGET]'
    }

    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)

    # Also handle any unicode escape sequences that might be problematic
    content = re.sub(r'\\U[0-9a-fA-F]{8}', '[EMOJI]', content)
    content = re.sub(r'\\u[0-9a-fA-F]{4}', '[EMOJI]', content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed emojis in {filename}")

if __name__ == "__main__":
    fix_emojis("mlb_hr_clean_v4.py")