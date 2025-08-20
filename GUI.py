import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit,
                             QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# Import our mathematical equation parser
try:
    from equation_parser import MathEquationParser
except ImportError:
    print("Warning: equation_parser module not found. Using basic evaluation.")
    MathEquationParser = None

class InteractiveStatisticsPlot(FigureCanvas):
    """
    Custom matplotlib canvas that handles interactive data point manipulation.
    This class demonstrates how event-driven programming can create responsive
    statistical visualizations.
    """
    
    def __init__(self, parent=None, statistics_update_callback=None):
        # Create matplotlib figure and axis
        self.figure = Figure(figsize=(10, 8))
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

        # Store the callback function - this is our reliable communication channel
        self.statistics_update_callback = statistics_update_callback
        
        # Data storage - using lists for easy manipulation
        self.x_data = []
        self.y_data = []
        
        # Interaction state variables
        self.dragging = False
        self.drag_point_index = None
        self.click_tolerance = 0.1  # How close click must be to select point
        
        # Connect mouse events - this is where the interactivity magic happens
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Initialize the plot
        self.setup_plot()
        self.update_plot()
        
    def setup_plot(self):
        """Configure the plot appearance and behavior."""
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-10, 10)
        self.axes.grid(True, alpha=0.3)
        self.axes.set_xlabel('X Values', fontsize=12)
        self.axes.set_ylabel('Y Values', fontsize=12)
        self.axes.set_title('Interactive Data Explorer - Click to add, Drag to move points', fontsize=14)
        
    def find_nearest_point(self, x, y):
        """
        Find the closest data point to the clicked location.
        This geometric calculation is crucial for smooth interaction.
        """
        if len(self.x_data) == 0:
            return None
            
        # Convert click coordinates to data coordinates
        distances = []
        for i in range(len(self.x_data)):
            # Calculate Euclidean distance - fundamental concept in statistics
            dist = np.sqrt((self.x_data[i] - x)**2 + (self.y_data[i] - y)**2)
            distances.append(dist)
        
        min_distance = min(distances)
        if min_distance < self.click_tolerance:
            return distances.index(min_distance)
        return None
    
    def on_mouse_press(self, event):
        """Handle mouse press events for point selection and creation."""
        if event.inaxes != self.axes:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Try to find existing point near click
        point_index = self.find_nearest_point(x, y)
        
        if point_index is not None:
            # Start dragging existing point
            self.dragging = True
            self.drag_point_index = point_index
        else:
            # Create new point - this demonstrates how data collection affects statistics
            self.x_data.append(x)
            self.y_data.append(y)
            self.update_plot()
            self.update_statistics()
    
    def on_mouse_release(self, event):
        """Handle mouse release to end dragging operations."""
        if self.dragging:
            self.dragging = False
            self.drag_point_index = None
            self.update_statistics()  # Recalculate stats after point movement
    
    def on_mouse_move(self, event):
        """
        Handle mouse movement during dragging.
        This is where you'll see real-time statistical changes as points move.
        """
        if not self.dragging or event.inaxes != self.axes:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None or self.drag_point_index is None:
            return
            
        # Update point position
        self.x_data[self.drag_point_index] = x
        self.y_data[self.drag_point_index] = y
        
        # Update plot in real-time
        self.update_plot()
    
    def update_plot(self):
        """Refresh the visualization with current data."""
        self.axes.clear()
        self.setup_plot()
        
        if len(self.x_data) > 0:
            # Plot points with larger size for easier interaction
            self.axes.scatter(self.x_data, self.y_data, s=100, c='blue', alpha=0.7, picker=True)
            
            # Add regression line if we have enough points
            if len(self.x_data) > 1:
                # Calculate and display linear regression - shows relationship strength
                coeffs = np.polyfit(self.x_data, self.y_data, 1)
                line_x = np.array([min(self.x_data), max(self.x_data)])
                line_y = coeffs[0] * line_x + coeffs[1]
                self.axes.plot(line_x, line_y, 'r--', alpha=0.8, label=f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
                self.axes.legend()
        
        self.draw()
    
    def update_statistics(self):
        """
        Trigger statistics update using our reliable callback function.
        This approach ensures we can always reach the main window's statistics method.
        """
        if self.statistics_update_callback:
            self.statistics_update_callback()
        else:
            print("Warning: No statistics callback function provided")

        
    def get_current_data(self):
        """Return current dataset as pandas DataFrame for easy export."""
        if len(self.x_data) > 0 and len(self.y_data) > 0:
            return pd.DataFrame({
                'x': self.x_data,
                'y': self.y_data
            })
        return pd.DataFrame()
    
    def load_data_from_csv(self, filename):
        """
        Load data from CSV file.
        This demonstrates how real-world data integrates with statistical analysis.
        """
        try:
            df = pd.read_csv(filename)
            if len(df.columns) < 2:
                raise ValueError("CSV must have at least 2 columns")
            
            # Use first two columns as x and y
            self.x_data = df.iloc[:, 0].tolist()
            self.y_data = df.iloc[:, 1].tolist()
            
            self.update_plot()
            self.update_statistics()
            return True
        except Exception as e:
            QMessageBox.warning(self.parent(), "Error", f"Could not load CSV: {str(e)}")
            return False
    
    def generate_data_from_equation(self, equation, x_range, y_noise, x_noise, num_points):
        """
        Generate data points based on mathematical equation with controlled noise.
        This is perfect for exploring how noise affects statistical measures.
        
        The key insight here is that we're creating a controlled experiment where
        we know the true underlying relationship, but we're adding measurement
        error to simulate real-world data collection challenges.
        """
        try:
            # Generate x values within specified range
            x_min, x_max = x_range
            base_x = np.linspace(x_min, x_max, num_points)
            
            # Add noise to x values if specified
            # This simulates measurement error in the independent variable
            if x_noise > 0:
                x_values = base_x + np.random.normal(0, x_noise, num_points)
            else:
                x_values = base_x
            
            # Use our sophisticated equation parser if available
            if MathEquationParser:
                parser = MathEquationParser()
                
                # Validate equation first
                is_valid, error_msg = parser.validate_equation(equation)
                if not is_valid:
                    raise ValueError(f"Invalid equation: {error_msg}")
                
                # Generate y values using the parser
                y_values = parser.parse_and_evaluate(equation, x_values)
                
            else:
                # Fallback to basic evaluation if parser not available
                y_values = []
                for x_val in x_values:
                    eq = equation.replace('x', str(x_val))
                    try:
                        y_val = eval(eq)
                        y_values.append(y_val)
                    except:
                        y_values.append(0)
                y_values = np.array(y_values)
            
            # Add noise to y values - this simulates measurement error in dependent variable
            # This is crucial for understanding how noise affects correlation and other measures
            if y_noise > 0:
                y_values += np.random.normal(0, y_noise, num_points)
            
            # Store the generated data
            self.x_data = x_values.tolist()
            self.y_data = y_values.tolist()
            
            # Critical fix: Ensure statistics are updated after data generation
            self.update_plot()
            self.update_statistics()  # This was missing - explains why stats weren't showing!
            return True
            
        except Exception as e:
            QMessageBox.warning(self.parent(), "Error", f"Could not generate data: {str(e)}")
            return False
    
    def clear_data(self):
        """Remove all data points."""
        self.x_data = []
        self.y_data = []
        self.update_plot()
        self.update_statistics()

class StatisticsCalculator:
    """
    Handles all statistical calculations.
    This class demonstrates the mathematical relationships you're exploring.
    """
    
    @staticmethod
    def calculate_all_statistics(x_data, y_data):
        """
        Calculate comprehensive statistics for the dataset.
        Each calculation reveals different aspects of data distribution and relationships.
        """
        if len(x_data) == 0 or len(y_data) == 0:
            return {}
        
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        stats = {}
        
        # Basic descriptive statistics for X
        stats['x_mean'] = np.mean(x_array)
        stats['x_median'] = np.median(x_array)
        stats['x_std'] = np.std(x_array, ddof=1) if len(x_array) > 1 else 0
        stats['x_variance'] = np.var(x_array, ddof=1) if len(x_array) > 1 else 0
        
        # Basic descriptive statistics for Y
        stats['y_mean'] = np.mean(y_array)
        stats['y_median'] = np.median(y_array)
        stats['y_std'] = np.std(y_array, ddof=1) if len(y_array) > 1 else 0
        stats['y_variance'] = np.var(y_array, ddof=1) if len(y_array) > 1 else 0
        
        # Relationship measures - these show how x and y variables relate
        if len(x_array) > 1:
            # Covariance measures how x and y vary together
            stats['covariance'] = np.cov(x_array, y_array)[0, 1]
            
            # Correlation coefficient normalizes covariance (-1 to 1 scale)
            correlation_matrix = np.corrcoef(x_array, y_array)
            stats['correlation'] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
        else:
            stats['covariance'] = 0
            stats['correlation'] = 0
        
        # Dataset size
        stats['count'] = len(x_data)
        
        return stats

class MainWindow(QMainWindow):
    """
    Main application window that orchestrates the interactive learning experience.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Statistics Explorer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create plot widget (left side)
        self.plot_widget = InteractiveStatisticsPlot(self, statistics_update_callback=self.update_statistics_display)
        main_layout.addWidget(self.plot_widget, 2)  # Takes 2/3 of space
        
        # Create control panel (right side)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)  # Takes 1/3 of space
        
        # Initialize statistics display
        self.update_statistics_display()
    
    def create_control_panel(self):
        """Create the control panel with all interactive elements."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File operations group
        file_group = QGroupBox("Data Import")
        file_layout = QVBoxLayout(file_group)
        
        csv_button = QPushButton("Load CSV File")
        csv_button.clicked.connect(self.load_csv_file)
        file_layout.addWidget(csv_button)
        
        save_button = QPushButton("Save Data to CSV")
        save_button.clicked.connect(self.save_csv_file)
        file_layout.addWidget(save_button)
        
        clear_button = QPushButton("Clear All Data")
        clear_button.clicked.connect(self.plot_widget.clear_data)
        file_layout.addWidget(clear_button)
        
        layout.addWidget(file_group)
        
        # Data generation group
        gen_group = QGroupBox("Generate Data")
        gen_layout = QFormLayout(gen_group)
        
        self.equation_input = QLineEdit("2*x + 1")
        gen_layout.addRow("Equation (use 'x'):", self.equation_input)
        
        self.x_min_input = QDoubleSpinBox()
        self.x_min_input.setRange(-100, 100)
        self.x_min_input.setValue(-5)
        gen_layout.addRow("X Min:", self.x_min_input)
        
        self.x_max_input = QDoubleSpinBox()
        self.x_max_input.setRange(-100, 100)
        self.x_max_input.setValue(5)
        gen_layout.addRow("X Max:", self.x_max_input)
        
        self.x_noise_input = QDoubleSpinBox()
        self.x_noise_input.setRange(0, 10)
        self.x_noise_input.setValue(0)
        self.x_noise_input.setSingleStep(0.1)
        gen_layout.addRow("X Noise (σ):", self.x_noise_input)
        
        self.y_noise_input = QDoubleSpinBox()
        self.y_noise_input.setRange(0, 10)
        self.y_noise_input.setValue(0.5)
        self.y_noise_input.setSingleStep(0.1)
        gen_layout.addRow("Y Noise (σ):", self.y_noise_input)
        
        self.num_points_input = QSpinBox()
        self.num_points_input.setRange(5, 1000)
        self.num_points_input.setValue(50)
        gen_layout.addRow("Number of Points:", self.num_points_input)
        
        generate_button = QPushButton("Generate Data")
        generate_button.clicked.connect(self.generate_data)
        gen_layout.addRow(generate_button)
        
        layout.addWidget(gen_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.statistics_display = QTextEdit()
        self.statistics_display.setReadOnly(True)
        self.statistics_display.setMaximumHeight(300)
        stats_layout.addWidget(self.statistics_display)
        
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def load_csv_file(self):
        """Handle CSV file loading."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV files (*.csv)")
        if filename:
            self.plot_widget.load_data_from_csv(filename)
            self.update_statistics_display()
    
    def save_csv_file(self):
        """Save current data to CSV file."""
        data_df = self.plot_widget.get_current_data()
        if data_df.empty:
            QMessageBox.information(self, "No Data", "No data points to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "data_export.csv", "CSV files (*.csv)")
        if filename:
            try:
                data_df.to_csv(filename, index=False)
                QMessageBox.information(self, "Success", f"Data saved successfully to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save CSV: {str(e)}")
    
    def generate_data(self):
        """Generate data based on user inputs."""
        equation = self.equation_input.text()
        x_range = (self.x_min_input.value(), self.x_max_input.value())
        x_noise = self.x_noise_input.value()
        y_noise = self.y_noise_input.value()
        num_points = self.num_points_input.value()
        
        self.plot_widget.generate_data_from_equation(equation, x_range, y_noise, x_noise, num_points)
        self.update_statistics_display()
    
    def update_statistics_display(self):
        """
        Update the statistics display with current calculations.
        This is where you'll see how data changes affect statistical measures.
        """
        stats = StatisticsCalculator.calculate_all_statistics(
            self.plot_widget.x_data, self.plot_widget.y_data
        )
        
        if not stats:
            self.statistics_display.setText("No data points available.\nClick on the plot to add points!")
            return
        
        # Format statistics for display
        display_text = f"""DATASET STATISTICS
{'='*30}

Data Points: {stats['count']}

X VARIABLE STATISTICS:
Mean: {stats['x_mean']:.4f}
Median: {stats['x_median']:.4f}
Standard Deviation: {stats['x_std']:.4f}
Variance: {stats['x_variance']:.4f}

Y VARIABLE STATISTICS:
Mean: {stats['y_mean']:.4f}
Median: {stats['y_median']:.4f}
Standard Deviation: {stats['y_std']:.4f}
Variance: {stats['y_variance']:.4f}

RELATIONSHIP MEASURES:
Covariance: {stats['covariance']:.4f}
Correlation Coefficient: {stats['correlation']:.4f}
"""
        
        self.statistics_display.setText(display_text)

