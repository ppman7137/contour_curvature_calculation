# -*- coding: utf-8 -*-
"""
Enhanced version of curvature calculation with interactive visualization
and correlation analysis.

Original author: curvature calculation by deepseek with parallelization
Enhanced by: Claude with interactive features and correlation analysis
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PointData:
    """Data class to hold point information"""
    x: np.ndarray
    y: np.ndarray
    epe: np.ndarray

    @property
    def points(self) -> np.ndarray:
        return np.column_stack((self.x, self.y))

# [Previous functions remain the same until create_interactive_plot]
# Include all previous functions here: generate_ellipse_points, read_file, write_file,
# fit_circle, calculate_curvature_for_segment, curvature_circle_fitting_closed_n_signed,
# and curvature_parametric_signed exactly as they were in the previous version

def generate_ellipse_points(n_points: int, noise: float) -> Optional[np.ndarray]:
    """Generate ellipse points with noise"""
    try:
        t = np.linspace(0, 2 * np.pi, n_points)
        a, b = 5, 3  # semi-major and semi-minor axes
        x = a * np.cos(t) + np.random.normal(0, noise, n_points)
        y = b * np.sin(t) + np.random.normal(0, noise, n_points)
        return np.column_stack((x, y))
    except Exception as e:
        logger.error(f"Error generating ellipse points: {e}")
        return None

def read_file(filename: str) -> Optional[PointData]:
    """Read data from CSV file and return PointData object"""
    try:
        #filepath = os.path.join(os.getcwd(), filename)  # Use the script's directory instead of current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        df = pd.read_csv(filepath)
        required_columns = {'X', 'Y', 'EPE'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"File must contain columns: {required_columns}")
        return PointData(
            x=df['X'].to_numpy(),
            y=df['Y'].to_numpy(),
            epe=df['EPE'].to_numpy()
        )
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None

def write_file(filename: str, point_data: PointData, 
               radius_1: np.ndarray, radius_2: np.ndarray) -> None:
    """Write results to CSV file"""
    try:
        # filepath = os.path.join(os.getcwd(), filename) # Use the script's directory instead of current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        df = pd.DataFrame({
            'Xctr': point_data.x,
            'Yctr': point_data.y,
            'EPE': point_data.epe,
            'radius_circle': radius_1,
            'radius_parametric': radius_2
        })
        df.to_csv(filepath, index=False)
        logger.info(f"File successfully written to {filepath}")
    except Exception as e:
        logger.error(f"Error writing file: {e}")

def fit_circle(points_subset: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a circle to a subset of points using least-squares"""
    if len(points_subset) < 3:
        raise ValueError("At least 3 points are required for circle fitting")
    
    A = np.column_stack((2 * points_subset[:, 0], 
                        2 * points_subset[:, 1], 
                        np.ones(len(points_subset))))
    b = np.sum(points_subset**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    center = x[:2]
    radius = np.sqrt(x[2] + np.sum(center**2))
    return center, radius

def calculate_curvature_for_segment(args: tuple) -> float:
    """Calculate curvature for a single segment"""
    i, points, n, num_points = args
    try:
        indices = [(i + j) % num_points for j in range(-n // 2 + 1, n // 2 + 1)]
        points_subset = points[indices]
        
        center, radius = fit_circle(points_subset)
        
        # Compute signed curvature using determinant
        x1, y1 = points_subset[0]
        x2, y2 = points_subset[len(points_subset) // 2]
        x3, y3 = points_subset[-1]
        det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        
        return (1 / radius) * np.sign(det)
    except Exception as e:
        logger.error(f"Error in circle fitting for segment {i}: {e}")
        return 0

@lru_cache(maxsize=None)
def curvature_circle_fitting_closed_n_signed(points_tuple: tuple, n: int) -> np.ndarray:
    """Parallelized curvature calculation using circle fitting"""
    points = np.array(points_tuple)
    num_points = len(points)
    args = [(i, points, n, num_points) for i in range(num_points)]
    
    try:
        with Pool(cpu_count()) as pool:
            curvatures = pool.map(calculate_curvature_for_segment, args)
        return np.array(curvatures)
    except KeyboardInterrupt:
        logger.warning("Calculation interrupted by user")
        return np.zeros(num_points)
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        return np.zeros(num_points)

def curvature_parametric_signed(points: np.ndarray, 
                              smoothing_param: float = 1.0) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Calculate signed curvature using parametric method"""
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing_param)
        x_spline, y_spline = splev(np.linspace(0, 1, 200), tck)
        x_deriv, y_deriv = splev(u, tck, der=1)
        x_deriv2, y_deriv2 = splev(u, tck, der=2)

        numerator = x_deriv * y_deriv2 - y_deriv * x_deriv2
        denominator = (x_deriv**2 + y_deriv**2)**(3/2)
        curvature = numerator / denominator

        return curvature, (x_spline, y_spline)
    except Exception as e:
        logger.error(f"Error in parametric curvature calculation: {e}")
        return np.zeros(len(points)), (np.array([]), np.array([]))


def calculate_regression_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Calculate linear regression coefficients and R² value"""
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    r2 = r2_score(y, model.predict(x))
    return model.coef_[0], model.intercept_, r2

def create_interactive_plot(point_data: PointData, 
                          curvatures_circle: np.ndarray,
                          curvatures_param: np.ndarray,
                          spline_data: Tuple[np.ndarray, np.ndarray]) -> plt.Figure:
    """Create interactive visualization with linked views and correlation analysis"""
    x_spline, y_spline = spline_data
    fig = plt.figure(figsize=(18, 12))  # Increased height for additional plot
    
    # Create GridSpec with more space between rows
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.6, wspace=0.3)

    # Scatter plot with spline (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(point_data.x, point_data.y, c=point_data.epe, 
                         cmap='viridis', label='Points', picker=True, pickradius=5)
    ax1.plot(x_spline, y_spline, '-', label='Spline', color='orange')
    ax1.set_title("Contour Points and Fitted Spline")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax1, label='EPE Values')
    ax1.legend()

    # Curvature comparison plot (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    arc_indices = np.arange(len(curvatures_circle))
    circle_plot = ax2.plot(arc_indices, curvatures_circle, 'o-', 
                          label='Circle Fitting', color='red')[0]
    param_plot = ax2.plot(arc_indices, curvatures_param, 'x-', 
                         label='Parametric', color='green')[0]
    highlight_point = ax2.scatter([], [], color='blue', s=100, zorder=5)
    ax2.set_title("Curvature Comparison")
    ax2.set_xlabel("Point Index")
    ax2.set_ylabel("Signed Curvature")
    ax2.legend()

    # Difference plot (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    differences = np.abs(curvatures_circle - curvatures_param)
    ax3.bar(arc_indices, differences, color='purple', alpha=0.6)
    ax3.set_title("Curvature Difference")
    ax3.set_xlabel("Point Index")
    ax3.set_ylabel("Absolute Difference")

    # Correlation plot (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate regression metrics for both methods
    slope_circle, intercept_circle, r2_circle = calculate_regression_metrics(
        curvatures_circle, point_data.epe)
    slope_param, intercept_param, r2_param = calculate_regression_metrics(
        curvatures_param, point_data.epe)

    # Plot correlation data
    ax4.scatter(curvatures_circle, point_data.epe, 
                label=f'Circle Fitting (R² = {r2_circle:.3f})', 
                color='red', alpha=0.5)
    ax4.scatter(curvatures_param, point_data.epe, 
                label=f'Parametric (R² = {r2_param:.3f})', 
                color='green', alpha=0.5)

    # Plot regression lines
    curvature_range = np.array([
        min(min(curvatures_circle), min(curvatures_param)),
        max(max(curvatures_circle), max(curvatures_param))
    ])
    ax4.plot(curvature_range, 
             slope_circle * curvature_range + intercept_circle, 
             '--', color='red', alpha=0.8)
    ax4.plot(curvature_range, 
             slope_param * curvature_range + intercept_param, 
             '--', color='green', alpha=0.8)

    ax4.set_title("EPE vs Curvature Correlation")
    ax4.set_xlabel("Curvature")
    ax4.set_ylabel("EPE")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Create annotations
    def create_annotation(ax):
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        return annot

    annot1 = create_annotation(ax1)
    annot2 = create_annotation(ax2)
    annot4 = create_annotation(ax4)

    def update_annotations(idx):
        # Update scatter plot annotation
        pos = (point_data.x[idx], point_data.y[idx])
        annot1.xy = pos
        annot1.set_text(f"Point {idx}\nX: {pos[0]:.2f}\nY: {pos[1]:.2f}\n"
                       f"EPE: {point_data.epe[idx]:.2f}")
        annot1.set_visible(True)

        # Update curvature plot annotation
        highlight_point.set_offsets([[idx, curvatures_circle[idx]]])
        annot2.xy = (idx, curvatures_circle[idx])
        annot2.set_text(f"Index: {idx}\nCircle: {curvatures_circle[idx]:.3f}\n"
                       f"Param: {curvatures_param[idx]:.3f}")
        annot2.set_visible(True)

        # Update correlation plot annotation
        annot4.xy = (curvatures_circle[idx], point_data.epe[idx])
        annot4.set_text(f"Point {idx}\nEPE: {point_data.epe[idx]:.2f}\n"
                       f"Circle Curv: {curvatures_circle[idx]:.3f}\n"
                       f"Param Curv: {curvatures_param[idx]:.3f}")
        annot4.set_visible(True)

    def on_pick(event):
        if event.artist == scatter:
            idx = event.ind[0]
            update_annotations(idx)
            fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes == ax2 and circle_plot.contains(event)[0]:
            idx = int(round(event.xdata))
            if 0 <= idx < len(curvatures_circle):
                update_annotations(idx)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the top margin
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])  # Adjust margins all around
    return fig

def main():
    """Main execution function"""
    # File handling
    input_file = 'slot_test.csv'
    output_file = 'output_slot_enhanced.csv'
    
    # Read input data
    point_data = read_file(input_file)
    if point_data is None:
        logger.error("Failed to read input data")
        return

    # Calculate curvatures
    n_points = 3  # Points for circle fitting
    logger.info("Calculating circle fitting curvature...")
    curvatures_circle = curvature_circle_fitting_closed_n_signed(
        tuple(map(tuple, point_data.points)), n_points)
    
    logger.info("Calculating parametric curvature...")
    curvatures_param, spline_data = curvature_parametric_signed(point_data.points)

    # Align parametric curvatures with circle fitting points
    curvatures_param_aligned = np.interp(
        np.linspace(0, 1, len(curvatures_circle)),
        np.linspace(0, 1, len(curvatures_param)),
        curvatures_param
    )

    # Save results
    write_file(output_file, point_data, curvatures_circle, curvatures_param_aligned)

    # Create and show interactive visualization
    fig = create_interactive_plot(point_data, curvatures_circle, 
                                curvatures_param_aligned, spline_data)
    plt.show()

if __name__ == "__main__":
    main()