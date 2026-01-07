import os
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorsys
import random

def extract_squarenut_range(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Str) and key.s == 'SquareNut':
                            if isinstance(value, ast.List) and len(value.elts) == 2:
                                x_range = ast.literal_eval(value.elts[0])
                                y_range = ast.literal_eval(value.elts[1])
                                return x_range, y_range
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    return None, None

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Slight variation in saturation
        lightness = 0.4 + (i % 2) * 0.1   # Slight variation in lightness
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors

def plot_ranges(directory=None):
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SquareNut Ranges')

    # Get list of files and generate colors
    files = [f for f in os.listdir(directory) if f.endswith('.py')]
    colors = generate_distinct_colors(len(files))

    # List to keep track of all label positions and bounding boxes
    label_positions = []
    bounding_boxes = []

    for filename, color in zip(files, colors):
        file_path = os.path.join(directory, filename)
        x_range, y_range = extract_squarenut_range(file_path)
        
        if x_range and y_range:
            width = x_range[1] - x_range[0]
            height = y_range[1] - y_range[0]
            
            # Check for overlap and apply x-jitter if necessary
            x_jitter = 0
            for existing_box in bounding_boxes:
                if (existing_box[0] <= x_range[0] <= existing_box[1] or 
                    existing_box[0] <= x_range[1] <= existing_box[1] or
                    x_range[0] <= existing_box[0] <= x_range[1]) and \
                   (existing_box[2] <= y_range[0] <= existing_box[3] or 
                    existing_box[2] <= y_range[1] <= existing_box[3] or
                    y_range[0] <= existing_box[2] <= y_range[1]):
                    x_jitter = random.uniform(-0.01, 0.01)
                    break
            
            # Apply the x-jitter to the bounding box
            x_range = (x_range[0] + x_jitter, x_range[1] + x_jitter)
            
            rect = Rectangle((x_range[0], y_range[0]), width, height, 
                             fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Add the new bounding box to the list
            bounding_boxes.append((x_range[0], x_range[1], y_range[0], y_range[1]))
            
            # Initial label position
            label_x = x_range[1]
            label_y = y_range[1]
            
            # Apply iterative jitter to labels
            jitter_attempts = 0
            while jitter_attempts < 50:  # Limit attempts to prevent infinite loop
                overlap = False
                for pos in label_positions:
                    if abs(pos[0] - label_x) < 0.03 and abs(pos[1] - label_y) < 0.03:
                        overlap = True
                        break
                
                if not overlap:
                    break
                
                # If overlap, apply additional jitter
                label_x += random.uniform(0.01, 0.03) * (1 if random.random() > 0.5 else -1)
                label_y += random.uniform(0.01, 0.03) * (1 if random.random() > 0.5 else -1)
                jitter_attempts += 1
            
            label_positions.append((label_x, label_y))
            
            ax.text(label_x, label_y, filename, fontsize=8, 
                    color=color, fontweight='bold',
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        else:
            print(f"Skipping file {filename}: No SquareNut field found or invalid format")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_ranges(os.path.dirname(os.path.abspath(__file__)))
