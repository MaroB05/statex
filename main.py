from GUI import *

def main():
    """Launch the interactive statistics explorer."""
    app = QApplication(sys.argv)
    
    # Set application style for better appearance
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
