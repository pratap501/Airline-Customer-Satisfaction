from src.features.build_features import build_feature_pipeline, load_and_prepare_data

X, y = load_and_prepare_data("data/Invistico_Airline.csv")

pipeline = build_feature_pipeline()
X_transformed = pipeline.fit_transform(X)

print("Feature matrix shape:", X_transformed.shape)
print("Target shape:", y.shape)
