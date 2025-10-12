import tkinter as tk
import sys
from gui import MANETSimulatorGUI
from simulator import MANETSimulator

def run_test_suite():
    """Run the test suite independently"""
    print("Starting MANET Simulator Test Suite...")
    simulator = MANETSimulator()
    results = simulator.run_test_suite()
    return results

def main():
    """Main function to run the MANET simulator"""
    # Check if test suite should be run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_suite()
        return
    
    try:
        # Run GUI application
        root = tk.Tk()
        app = MANETSimulatorGUI(root)
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print("Application error:", e)
            raise
            
    except ImportError as e:
        print("Error importing modules:", e)
        print("Make sure all required modules are in the same directory.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
