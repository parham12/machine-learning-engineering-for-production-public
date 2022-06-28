from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from main import clf


# tssesttttt

def test_pipeline_and_scaler():

    # Check if clf is an instance of sklearn.pipeline.Pipeline 
    isPipeline = isinstance(clf, Pipeline)
    assert isPipeline
    
    if isPipeline:
        # Check if first step of pipeline is an instance of 
        # sklearn.preprocessing.StandardScaler
        firstStep = [v for v in clf.named_steps.values()][0]
        assert isinstance(firstStep, StandardScaler)
