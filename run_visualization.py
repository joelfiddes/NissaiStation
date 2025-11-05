#!/usr/bin/env python3
"""
Meteorological Data Visualization Launcher
This script helps you generate both static and interactive visualizations
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("🌤️  METEOROLOGICAL DATA VISUALIZATION SUITE")
    print("=" * 60)
    print()
    
    # Check if the CSV file exists
    csv_file = "combined_meteodata2025.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found in current directory")
        print("Please ensure the CSV file is in the same directory as this script.")
        return
    
    print(f"📊 Found data file: {csv_file}")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Generate static PDF report (all variables in panels)")
        print("2. Launch interactive dashboard (selectable parameters)")
        print("3. Generate both")
        print("4. Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🔄 Generating static PDF report...")
            try:
                result = subprocess.run([sys.executable, "generate_static_pdf.py"], 
                                      capture_output=True, text=True, cwd=".")
                if result.returncode == 0:
                    print("✅ Static PDF report generated successfully!")
                    print("Output:", result.stdout)
                else:
                    print("❌ Error generating PDF:")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "2":
            print("\n🚀 Launching interactive dashboard...")
            print("📖 Instructions:")
            print("   - Your browser will open automatically at http://127.0.0.1:8050")
            print("   - Use the dropdowns to select variable categories and specific variables")
            print("   - You can overlay multiple variables on the same plot")
            print("   - Choose different plot types (time series, scatter, distribution)")
            print("   - Adjust the date range as needed")
            print("   - Press Ctrl+C to stop the dashboard")
            print()
            
            try:
                subprocess.run([sys.executable, "interactive_meteo_plot.py"], cwd=".")
            except KeyboardInterrupt:
                print("\n🛑 Interactive dashboard stopped.")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == "3":
            print("\n🔄 Generating static PDF report first...")
            try:
                result = subprocess.run([sys.executable, "generate_static_pdf.py"], 
                                      capture_output=True, text=True, cwd=".")
                if result.returncode == 0:
                    print("✅ Static PDF report generated successfully!")
                    print()
                    
                    print("🚀 Now launching interactive dashboard...")
                    print("📖 Instructions:")
                    print("   - Your browser will open automatically at http://127.0.0.1:8050")
                    print("   - Use the dropdowns to select variable categories and specific variables")
                    print("   - You can overlay multiple variables on the same plot")
                    print("   - Choose different plot types (time series, scatter, distribution)")
                    print("   - Adjust the date range as needed")
                    print("   - Press Ctrl+C to stop the dashboard")
                    print()
                    
                    subprocess.run([sys.executable, "interactive_meteo_plot.py"], cwd=".")
                else:
                    print("❌ Error generating PDF:")
                    print(result.stderr)
            except KeyboardInterrupt:
                print("\n🛑 Interactive dashboard stopped.")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()