Create a clean, single-page "Breast Cancer Risk Predictor" form app.

Form fields (all required, number inputs, step "any"), with these exact JSON keys:
- worst_perimeter (placeholder 107.0)
- mean_concave_points (placeholder 0.05)
- area_error (placeholder 24.5)
- worst_concavity (placeholder 0.27)
- worst_compactness (placeholder 0.21)
- mean_compactness (placeholder 0.10)

On submit: POST JSON with those keys to https://YOUR-N8N-URL/webhook/predict
(Content-Type: application/json). Show a loading state.

The response looks like:
{ "prediction": "malignant" | "benign", "confidence": 0.97,
  "probability_malignant": 0.03, "probability_benign": 0.97, "warnings": [] }

Show the result in a card below the form: big label (red for malignant, green for benign),
confidence as a percentage, and any warnings in amber. Handle network errors with a friendly message.
Add a small disclaimer: "Educational demo, not medical advice."
