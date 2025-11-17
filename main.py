import pandas as pd
from src.preprocess_data import convert_data, decode_data, clean_and_save_data
from src.prepare_data import load_features_and_target
from src.split_data import split_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


RANDOM_SEED = 42
TEST_SIZE = 0.2

# === LOAD RAW DATA ===
df = pd.read_csv("personality_dataset.csv")

# === ENCODE DATA ===
encoders, encoded_df = convert_data(df)

# test decoding 
encoded_list = encoded_df.iloc[0]
decoded_values = decode_data(encoded_list, encoders)


# === SAVE CLEANED DATA ===
clean_and_save_data(df, "cleaned_dataset.csv")

# === LOAD CLEANED DATA ===
clean_df = pd.read_csv("cleaned_dataset.csv")
# === SEPARATE INPUTS (X) AND TARGET (y) ===
X, y = load_features_and_target(clean_df)

print(X.head())
print(y.head())

# === SPLIT DATA INTO TRAIN AND TEST ===
X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    random_seed=RANDOM_SEED,
    test_size=TEST_SIZE
)

print("Training and test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# === TRAIN DECISION TREE ===
model = DecisionTreeClassifier(random_state=RANDOM_SEED, criterion='entropy', max_depth=3)
model.fit(X_train, y_train)

print("Model trained!")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# --- Make predictions on the test set ---
y_pred = model.predict(X_test)


# --- Evaluation metrics ---

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)


print("\nMODEL EVALUATION RESULTS")
print("-------------------------")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

print("\nConfusion Matrix:")
print(cm)



def tune_model(X_train, y_train, X_test, y_test):
    max_depth_values = [3, 5, 10, None]
    min_samples_split_values = [2, 5, 10]
    criteria = ["gini", "entropy"]


    best_score = 0
    best_params = None


    for depth in max_depth_values:
        for split in min_samples_split_values:
            for crit in criteria:

                model = DecisionTreeClassifier(
                    random_state=42,
                    max_depth=depth,
                    min_samples_split=split,
                    criterion=crit
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")
                cm = confusion_matrix(y_test, y_pred)

                print(f"\nParameters: max_depth={depth}, min_samples_split={split}, criterion={crit}")
                print(f"Accuracy:     {acc:.3f}")
                print(f"F1 Score:     {f1:.3f}")
                print(f"Confusion:\n{cm}")

                if acc > best_score:
                    best_score = acc
                    best_params = (depth, split, crit)

    print("\n--- BEST PARAMETERS FOUND ---")
    print(f"Accuracy: {best_score:.3f}")
    print(f"max_depth={best_params[0]}, min_samples_split={best_params[1]}, criterion={best_params[2]}")

    return best_params


            
# best_depth, best_split, best_criterion = tune_model(X_train, y_train, X_test, y_test)



best_depth = None
best_split = 10
best_criterion = "gini"

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=best_depth,
    min_samples_split=best_split,
    criterion=best_criterion
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate final model
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\nFINAL MODEL RESULTS")
print("---------------------")
print(f"Accuracy:   {acc:.3f}")
print(f"F1 Score:   {f1:.3f}")
print("Confusion matrix:\n", cm)





