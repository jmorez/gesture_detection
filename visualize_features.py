import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def load_model_and_data():
    """Load the trained classifier and training data."""
    if not os.path.exists("gesture_classifier.pkl"):
        print("Error: No trained model found! Run learning mode first.")
        return None, None, None

    with open("gesture_classifier.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Handle both old and new format
    if isinstance(model_data, dict):
        clf = model_data["classifier"]
        X_train = model_data.get("X_train", None)
        y_train = model_data.get("y_train", None)
    else:
        # Old format - just the classifier
        clf = model_data
        X_train = None
        y_train = None

    return clf, X_train, y_train


def visualize_feature_space(clf, X_train, y_train):
    """Visualize the feature space using real training data."""

    # Feature names for reference
    feature_names = [
        "Left Hand Velocity",
        "Right Hand Velocity",
        "Left Hand Height (rel to shoulders)",
        "Right Hand Height (rel to shoulders)",
    ]

    # Class labels
    class_names = ["Waving", "Applauding", "Nothing"]
    colors = ["green", "orange", "gray"]

    # Get feature importances from the decision tree
    feature_importances = clf.feature_importances_

    # Determine if we have real data
    use_real_data = X_train is not None and y_train is not None

    # Create feature importance plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Feature Importances
    ax = axes[0, 0]
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0])
    ax.barh(pos, feature_importances[sorted_idx], align="center", color="skyblue")
    ax.set_yticks(pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance in Decision Tree")
    ax.grid(axis="x", alpha=0.3)

    # 2. Decision Tree Structure (text representation)
    ax = axes[0, 1]
    ax.axis("off")

    tree_info = f"Decision Tree Statistics:\n\n"
    tree_info += f"Max Depth: {clf.get_depth()}\n"
    tree_info += f"Number of Leaves: {clf.get_n_leaves()}\n"
    tree_info += f"Number of Features: {clf.n_features_in_}\n\n"
    tree_info += f"Classes: {', '.join(class_names)}\n"

    if use_real_data:
        tree_info += f"\nTraining Samples:\n"
        for i, name in enumerate(class_names):
            count = np.sum(y_train == i)
            tree_info += f"  {name}: {count}\n"

    tree_info += "\nMost Important Features:\n"
    top_3_idx = np.argsort(feature_importances)[-3:][::-1]
    for idx in top_3_idx:
        tree_info += f"  {feature_names[idx]}: {feature_importances[idx]:.3f}\n"

    ax.text(
        0.1,
        0.5,
        tree_info,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Model Information")

    # 3. Plot in speed space (real or synthetic data)
    ax = axes[1, 0]

    if use_real_data:
        # Use real training data
        for i, (color, name) in enumerate(zip(colors, class_names)):
            mask = y_train == i
            if np.any(mask):
                ax.scatter(
                    X_train[mask, 0],
                    X_train[mask, 1],
                    c=color,
                    label=name,
                    alpha=0.6,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
        ax.set_title("Feature Space: Hand Velocities (Real Data)")
    else:
        # Fallback to synthetic data if no real data available
        print("Warning: No training data found, using synthetic examples")
        np.random.seed(42)
        n_samples = 100

        # Generate synthetic velocity data
        waving_left_vel = np.random.uniform(15, 40, n_samples)
        waving_right_vel = np.random.uniform(15, 40, n_samples)

        applause_left_vel = np.random.uniform(20, 45, n_samples)
        applause_right_vel = np.random.uniform(20, 45, n_samples)

        nothing_left_vel = np.random.uniform(0, 8, n_samples)
        nothing_right_vel = np.random.uniform(0, 8, n_samples)

        ax.scatter(
            waving_left_vel,
            waving_right_vel,
            c="green",
            label="Waving",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            applause_left_vel,
            applause_right_vel,
            c="orange",
            label="Applauding",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            nothing_left_vel,
            nothing_right_vel,
            c="gray",
            label="Nothing",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_title("Feature Space: Hand Velocities (Synthetic Example)")

    ax.set_xlabel("Left Hand Velocity (pixels/frame)", fontsize=11)
    ax.set_ylabel("Right Hand Velocity (pixels/frame)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distance vs Combined Speed
    ax = axes[1, 1]

    if use_real_data:
        # Use real training data
        for i, (color, name) in enumerate(zip(colors, class_names)):
            mask = y_train == i
            if np.any(mask):
                # Feature 2: right_x, 3: right_y, 4: velocity_diff
                ax.scatter(
                    X_train[mask, 2],
                    X_train[mask, 3],
                    c=color,
                    label=name,
                    alpha=0.6,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
        ax.set_title("Feature Space: Hand Heights (Real Data)")
    else:
        # Fallback to synthetic data
        waving_left_height = np.random.uniform(-50, 50, n_samples)
        waving_right_height = np.random.uniform(-50, 50, n_samples)

        applause_left_height = np.random.uniform(0, 100, n_samples)
        applause_right_height = np.random.uniform(0, 100, n_samples)

        nothing_left_height = np.random.uniform(50, 150, n_samples)
        nothing_right_height = np.random.uniform(50, 150, n_samples)

        ax.scatter(
            waving_left_height,
            waving_right_height,
            c="green",
            label="Waving",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            applause_left_height,
            applause_right_height,
            c="orange",
            label="Applauding",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            nothing_left_height,
            nothing_right_height,
            c="gray",
            label="Nothing",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_title("Feature Space: Hand Heights (Synthetic Example)")

    ax.set_xlabel("Left Hand Height (rel to shoulders)", fontsize=11)
    ax.set_ylabel("Right Hand Height (rel to shoulders)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_space_visualization.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved as 'feature_space_visualization.png'")

    # Also save to assets folder if it exists
    if os.path.exists("assets"):
        plt.savefig(
            "assets/feature_space_visualization.png", dpi=150, bbox_inches="tight"
        )
        print("Also saved to 'assets/feature_space_visualization.png'")

    plt.show()


def visualize_3d_feature_space(clf, X_train, y_train):
    """Visualize the feature space in 3D using the three most important features."""

    feature_names = [
        "Left Hand Velocity",
        "Right Hand Velocity",
        "Left Hand Height (rel to shoulders)",
        "Right Hand Height (rel to shoulders)",
    ]

    class_names = ["Waving", "Applauding", "Nothing"]
    colors = ["green", "orange", "gray"]

    # Check if we have real data
    if X_train is None or y_train is None:
        print("\nNo training data available for 3D visualization")
        print("Please retrain your model to save training data")
        return

    # Get the three most important features
    feature_importances = clf.feature_importances_
    top_3_idx = np.argsort(feature_importances)[-3:][::-1]

    print(f"\n3D Visualization using top 3 features:")
    for i, idx in enumerate(top_3_idx):
        print(
            f"  {i+1}. {feature_names[idx]} (importance: {feature_importances[idx]:.3f})"
        )

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create mesh grid for decision boundaries (using only 3 features)
    # Get data ranges for the 3 most important features
    x_min, x_max = (
        X_train[:, top_3_idx[0]].min() - 10,
        X_train[:, top_3_idx[0]].max() + 10,
    )
    y_min, y_max = (
        X_train[:, top_3_idx[1]].min() - 10,
        X_train[:, top_3_idx[1]].max() + 10,
    )
    z_min, z_max = (
        X_train[:, top_3_idx[2]].min() - 10,
        X_train[:, top_3_idx[2]].max() + 10,
    )

    # Create a coarser grid for decision boundaries (for performance)
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
        np.linspace(z_min, z_max, 20),
    )

    # Prepare full feature vectors for prediction
    # We need to fill in the missing feature with mean values
    grid_samples = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Create full 4D feature vectors by inserting the 3 selected features
    full_features = np.zeros((grid_samples.shape[0], clf.n_features_in_))
    for i in range(clf.n_features_in_):
        if i in top_3_idx:
            # Use grid value for selected features
            idx_in_grid = np.where(top_3_idx == i)[0][0]
            full_features[:, i] = grid_samples[:, idx_in_grid]
        else:
            # Use mean value for non-selected features
            full_features[:, i] = X_train[:, i].mean()

    # Predict on grid
    Z = clf.predict(full_features)
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries as semi-transparent surfaces
    for class_val in range(len(class_names)):
        # Create isosurface for each class
        ax.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            Z[:, :, 0],
            levels=[class_val - 0.5, class_val + 0.5],
            colors=[colors[class_val]],
            alpha=0.1,
        )

    # Plot each class
    for i, (color, name) in enumerate(zip(colors, class_names)):
        mask = y_train == i
        if np.any(mask):
            ax.scatter(
                X_train[mask, top_3_idx[0]],
                X_train[mask, top_3_idx[1]],
                X_train[mask, top_3_idx[2]],
                c=color,
                label=name,
                alpha=0.8,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel(feature_names[top_3_idx[0]], fontsize=11, labelpad=10)
    ax.set_ylabel(feature_names[top_3_idx[1]], fontsize=11, labelpad=10)
    ax.set_zlabel(feature_names[top_3_idx[2]], fontsize=11, labelpad=10)
    ax.set_title(
        "3D Feature Space (Top 3 Most Important Features)", fontsize=14, pad=20
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig("feature_space_3d.png", dpi=150, bbox_inches="tight")
    print("3D visualization saved as 'feature_space_3d.png'")

    # Also save to assets folder if it exists
    if os.path.exists("assets"):
        plt.savefig("assets/feature_space_3d.png", dpi=150, bbox_inches="tight")
        print("Also saved to 'assets/feature_space_3d.png'")

    plt.show()


if __name__ == "__main__":
    print("=== Gesture Feature Space Visualization ===\n")

    clf, X_train, y_train = load_model_and_data()

    if clf is not None:
        print(f"Loaded classifier with {clf.n_features_in_} features")
        print(f"Number of classes: {clf.n_classes_}")

        if X_train is not None:
            print(f"Loaded {len(X_train)} training samples")
        else:
            print("No training data saved - retrain model to include data")

        # Visualize feature space
        visualize_feature_space(clf, X_train, y_train)

        # Visualize in 3D
        visualize_3d_feature_space(clf, X_train, y_train)

        print("\n✓ Visualization complete!")
    else:
        print("\nPlease train a model first using the learning mode in main.py")
