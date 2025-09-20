#!/usr/bin/env python3
# Quick run script for MLB Production System

import os
import sys
from pathlib import Path

def main():
    print("MLB Production System Launcher")
    print("=" * 40)
    print()
    print("Choose an option:")
    print("1. Run Production System (Daily Predictions)")
    print("2. Run v4 Lite System (Advanced Analytics)")
    print("3. Run v4 Dashboard (Interactive Interface)")
    print("4. Install v4 Dependencies")
    print("5. Exit")
    print()

    choice = input("Enter choice (1-5): ").strip()

    if choice == '1':
        print("\nRunning Production System...")
        os.system('python production_system.py')

    elif choice == '2':
        print("\nRunning v4 Lite System...")
        os.chdir('v4_advanced')
        os.system('python RBI_v4_lite.py')

    elif choice == '3':
        print("\nStarting v4 Dashboard...")
        os.chdir('v4_advanced')
        os.system('streamlit run dashboard_v4.py')

    elif choice == '4':
        print("\nInstalling v4 Dependencies...")
        os.chdir('v4_advanced')
        os.system('python install_v4.py')

    elif choice == '5':
        print("\nGoodbye!")
        sys.exit(0)

    else:
        print("\nInvalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
