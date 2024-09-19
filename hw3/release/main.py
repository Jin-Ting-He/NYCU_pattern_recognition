import pandas as pd
import numpy as np
from loguru import logger

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, compute_accuracy, plot_feature_importance
from src.decision_tree import entropy, gini_index


def main():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    """
    Feel free to modify the following section if you need.
    Remember to print out logs with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=1000,
        learning_rate=0.003,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)

    accuracy_ = compute_accuracy(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_pred_probs, y_test, fpath='./adaboost_roc_curve.png')
    feature_importance = clf_adaboost.compute_feature_importance()
    logger.info(f'Feature Importance: {feature_importance}')
    plot_feature_importance(feature_importance, feature_names, fpath='./adaboost_feature_importance.png')
    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=1000,
        learning_rate=0.003,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = compute_accuracy(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_pred_probs, y_test, fpath='./bagging_roc_curve.png')
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./bagging_feature_importance.png')
    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = compute_accuracy(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    feature_importance = clf_tree.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./decision_tree_feature_importance.png')
    y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    gini = gini_index(y)
    logger.info(f"Gini Index: {gini:.4f}")
    ent = entropy(y)
    logger.info(f"Entropy: {ent:.4f}")


if __name__ == '__main__':
    main()
