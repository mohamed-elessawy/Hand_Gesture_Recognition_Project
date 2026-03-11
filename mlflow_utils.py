import os
import mlflow
import mlflow.sklearn
import tempfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# This function will set up MLflow to save our model training runs and results in a local folder called "mlruns"
def setup_mlflow(experiment_name="Hand_Gesture_Recognition"):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)


def log_models(models_dict, metrics_dict, X_test, y_test):
    # Store the unique ID for each MLflow run here
    run_ids = {}

    # Loop through each model in our dictionary one by one
    for model_name, model in models_dict.items():
        
        # Get the matching scores for this specific model
        metrics = metrics_dict[model_name]

        # Start a new MLflow run and give it the name of the model
        with mlflow.start_run(run_name=model_name):

            # Log Parameters 
            # hasattr checks if the model has a function called "get_params"
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Log Metrics
            mlflow.log_metrics(metrics)

            # Log the Model itself so it is saved in the registry later
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Create and Log the Confusion Matrix Picture
            # Force the model to guess the gestures for our test data
            y_pred = model.predict(X_test)
            
            # Get a clean, alphabetical list of all our gesture names (ascending order)
            labels = sorted(set(y_test))
            
            # Calculate the confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            
            # Set up a picture canvas large enough to fit all 18 gesture names without squeezing
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw the grid on the canvas
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
                ax=ax, colorbar=False, xticks_rotation="vertical")
            
            ax.set_title(f"Confusion Matrix - {model_name}")
            fig.tight_layout()

            # Create a temporary folder on the computer to save and hold the picture
            with tempfile.TemporaryDirectory() as tmp_dir:
                cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
                fig.savefig(cm_path, dpi=120)
                
                # Upload the picture from the temporary folder into MLflow
                mlflow.log_artifact(cm_path, artifact_path="plots")

            # Close the picture so it doesn't show up in the notebook and waste memory
            plt.close(fig)

            # Save the run ID to the dictionary so we know exactly where this model is saved
            run_ids[model_name] = mlflow.active_run().info.run_id
            
    return run_ids