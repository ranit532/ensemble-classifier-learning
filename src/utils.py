import os
import joblib

def save_model(model, name, folder="models"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")

def save_plot(fig, name, folder="images"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved at {path}")
