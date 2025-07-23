import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, find_core_cong


def _plot_pie(input_df: pd.DataFrame, erlang: float, save_fp: str):
    for column in ['old_bandwidth', 'num_segments', 'longest_reach']:
        plt.figure(figsize=(6, 6), dpi=300)
        counts = input_df[column].value_counts()
        input_df[column].value_counts().plot(kind='pie', autopct=lambda p: f'{p:.1f}%',
                                             textprops={'color': 'white', 'weight': 'bold'})
        plt.title(f'Pie Chart for {column} - Erlang {erlang}', weight='bold')

        # Create custom labels for the legend
        labels = [f'{label}: {count:,}' for label, count in counts.items()]
        plt.legend(labels, loc='best')

        tmp_fp = os.path.join(save_fp, f'pie_chart_{column}_{erlang}.png')
        plt.savefig(tmp_fp, bbox_inches='tight')


def _plot_hist(erlang: float, save_fp: str, input_df: pd.DataFrame):
    for column in ['path_length', 'ave_cong']:
        plt.figure(figsize=(12, 6), dpi=300)

        plt.subplot(1, 2, 1)
        sns.histplot(input_df[column], kde=True)
        plt.title(f'Histogram for {column} - Erlang {erlang}', weight='bold')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sns.boxplot(x=input_df[column])
        plt.title(f'Box Plot for {column} - Erlang {erlang}', weight='bold')

        tmp_fp = os.path.join(save_fp, f'hist_box_{column}_{erlang}.png')
        plt.savefig(tmp_fp, bbox_inches='tight')


def plot_data(sim_dict: dict, input_df: pd.DataFrame, erlang: float):
    """
    Plots data related to machine learning simulation runs.

    :param sim_dict: Relevant information for every simulation.
    :param input_df: Relevant information for simulations as a dataframe.
    :param erlang: Traffic volume value.
    """
    save_fp = os.path.join('data', 'plots', sim_dict['train_file_path'], 'input_analysis')
    create_dir(file_path=save_fp)

    _plot_pie(erlang=erlang, input_df=input_df, save_fp=save_fp)
    _plot_hist(erlang=erlang, input_df=input_df, save_fp=save_fp)


def _get_ml_obs(tmp_dict: dict, engine_props: dict, sdn_props: object):
    df_processed = pd.DataFrame(tmp_dict, index=[0])
    df_processed = pd.get_dummies(df_processed, columns=['old_bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    for bandwidth, percent in engine_props['request_distribution'].items():
        if percent > 0:
            if bandwidth != sdn_props.bandwidth:
                df_processed[f'old_bandwidth_{bandwidth}'] = 0

    column_order_list = ['path_length', 'longest_reach', 'ave_cong', 'old_bandwidth_50',
                         'old_bandwidth_100', 'old_bandwidth_200', 'old_bandwidth_400']
    df_processed = df_processed.reindex(columns=column_order_list)

    return df_processed


def get_ml_obs(req_dict: dict, engine_props: dict, sdn_props: object):
    """
    Creates a single entry or observation structured properly for a machine learning model to make a prediction.

    :param req_dict: Holds request information.
    :param engine_props: Engine properties.
    :param sdn_props: SDN controller properties.
    :return: The correctly formatted observation.
    :rtype: pd.DataFrame
    """
    path_length = find_path_len(path_list=sdn_props.path_list, topology=engine_props['topology'])
    cong_arr = np.array([])
    for core_num in range(engine_props['cores_per_link']):
        curr_cong = find_core_cong(core_index=core_num, net_spec_dict=sdn_props.net_spec_dict,
                                   path_list=sdn_props.path_list)
        cong_arr = np.append(cong_arr, curr_cong)

    tmp_dict = {
        'old_bandwidth': req_dict['bandwidth'],
        'path_length': path_length,
        'longest_reach': req_dict['mod_formats']['QPSK']['max_length'],
        'ave_cong': float(np.mean(cong_arr)),
    }

    return _get_ml_obs(engine_props=engine_props, sdn_props=sdn_props, tmp_dict=tmp_dict)


def load_model(engine_props: dict):
    """
    Loads a trained machine learning model.

    :param engine_props: Properties from engine.
    :return: The trained model.
    """

    model_fp = os.path.join('logs', engine_props['ml_model'], engine_props['train_file_path'],
                            f"{engine_props['ml_model']}_{str(int(engine_props['erlang']))}.joblib")
    resp = joblib.load(filename=model_fp)

    return resp


def save_model(sim_dict: dict, model, algorithm: str, erlang: str):
    """
    Saves a trained machine learning model.

    :param sim_dict: The simulation dictionary.
    :param model: The trained model.
    :param algorithm: The filename to save the model as.
    :param erlang: The Erlang value.
    """
    base_fp = os.path.join('logs', algorithm, sim_dict['train_file_path'])
    create_dir(file_path=base_fp)

    save_fp = os.path.join(base_fp, f'{algorithm}_{erlang}.joblib')
    joblib.dump(model, save_fp)


def process_data(sim_dict: dict, input_df: pd.DataFrame, erlang: float):
    """
    Process data for machine learning model.

    :param sim_dict: Holds relevant simulation information.
    :param input_df: Input dataframe.
    :param erlang: Traffic volume.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    plot_data(input_df=input_df, erlang=erlang, sim_dict=sim_dict)
    df_processed = pd.get_dummies(input_df, columns=['old_bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    return df_processed


def even_process_data(input_df: pd.DataFrame, per_slice: bool, erlang: float, sim_dict: dict):
    """
    Process data for a machine learning model.

    :param input_df: Input dataframe.
    :param per_slice: Boolean flag to indicate if equal entries for each num_segments are required.
    :param sim_dict: Holds relevant simulation information.
    :param erlang: Current traffic volume.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    if per_slice:
        unique_segments = input_df['num_segments'].unique()
        dfs = [input_df[input_df['num_segments'] == segment] for segment in unique_segments]
        min_size = min(len(df) for df in dfs)
        sampled_dfs = [df.sample(n=min_size, random_state=42) for df in dfs]
        df_processed = pd.concat(sampled_dfs).sample(frac=1, random_state=42)
    else:
        df1 = input_df[input_df['num_segments'] == 1]
        df2 = input_df[input_df['num_segments'] == 2]
        df4 = input_df[input_df['num_segments'] == 4]
        df8 = input_df[input_df['num_segments'] == 8]

        min_size = min(len(df1), len(df2), len(df4), len(df8))

        df1 = df1.sample(n=int(min_size * 0.05), random_state=42)  # 10% of the smallest group size
        df2 = df2.sample(n=int(min_size * 0.35), random_state=42)  # 30% of the smallest group size
        df4 = df4.sample(n=int(min_size * 0.35), random_state=42)  # 30% of the smallest group size
        df8 = df8.sample(n=int(min_size * 0.25), random_state=42)  # 30% of the smallest group size

        df_processed = pd.concat([df1, df2, df4, df8]).sample(frac=1, random_state=42)

    return process_data(sim_dict=sim_dict, input_df=df_processed, erlang=erlang)


def plot_feature_importance(sim_dict: dict, model, feature_names_list: list, erlang: float, x_test: np.array,
                            y_test: np.array):
    """
    Plots the feature importance for a model.

    :param sim_dict: The simulation dictionary.
    :param model: Trained model.
    :param feature_names_list: List of feature names.
    :param erlang: The Erlang value.
    :param x_test: The test data.
    :param y_test: The test labels.
    """
    try:
        # Tree-based models
        importances = model.feature_importances_
    except AttributeError:
        try:
            # Logistic Regression
            importances = np.abs(model.coef_[0])
        except AttributeError:
            # KNN
            perm_importance = permutation_importance(model, x_test, y_test)
            importances = perm_importance.importances_mean

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.title("Feature Rankings", weight='bold')
    plt.bar(range(len(importances)), importances[indices],
            color="b", align="center")
    plt.xticks(range(len(importances)), [feature_names_list[i] for i in indices], rotation='vertical')
    plt.xlim([-1, len(importances)])

    save_fp = os.path.join('data', 'plots', sim_dict['train_file_path'])
    create_dir(file_path=save_fp)

    save_fp = os.path.join(save_fp, f'feature_rankings_{erlang}.png')
    plt.savefig(save_fp, bbox_inches='tight')


def _plot_confusion(y_test: np.array, y_pred: np.array, accuracy: float, precision: float, recall: float,
                    f_score: float, sim_dict: dict, erlang: str):
    # Calculate accuracy for each unique num_segments value
    unique_segments = np.unique(y_test)
    accuracy_per_segment = []
    for segment in unique_segments:
        mask = (y_test == segment)  # pylint: disable=superfluous-parens
        segment_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        accuracy_per_segment.append(f"NS={segment}: {segment_accuracy:.4f}")

    # Convert the list of accuracies to a single string
    accuracy_str = ', '.join(accuracy_per_segment)

    # Add accuracy, precision, recall, and F1 score to the plot
    plt.text(0.2, 1.1, f'Accuracy: {accuracy:.4f}, {accuracy_str}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.2, 1.2, f'Precision: {precision:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.2, 1.3, f'Recall: {recall:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.2, 1.4, f'F1 Score: {f_score:.4f}', fontsize=12, transform=plt.gca().transAxes)

    save_fp = os.path.join('data', 'plots', sim_dict['train_file_path'])
    create_dir(file_path=save_fp)

    save_fp = os.path.join(save_fp, f'confusion_matrix_{erlang}.png')
    plt.savefig(save_fp, bbox_inches='tight')


def plot_confusion(sim_dict: dict, y_test: np.array, y_pred: np.array, erlang: str, algorithm: str):
    """
    Plots a confusion matrix and prints out the accuracy, precision, recall, and F1 score.

    :param sim_dict: The simulation dictionary.
    :param y_test: Testing data.
    :param y_pred: Predictions.
    :param erlang: The Erlang value.
    :param algorithm: The algorithm used.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    labels = np.unique(np.concatenate((y_test, y_pred)))
    # Plot a confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8), dpi=300)  # Increase the quality by increasing dpi
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {algorithm}', weight='bold')
    plt.xlabel('Predicted Segments', weight='bold')
    plt.ylabel('Actual Segments', weight='bold')

    _plot_confusion(y_test=y_test, y_pred=y_pred, accuracy=accuracy, precision=precision, recall=recall,
                    f_score=f_score, sim_dict=sim_dict, erlang=erlang)


def plot_2d_clusters(df_pca: pd.DataFrame):
    """
    Plot the test data points and their predicted labels.

    :param df_pca: A dataframe normalized with PCA.
    """
    plt.figure(figsize=(10, 8))

    # Plot the test data points, colored by their predicted labels
    scatter = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["predicted_label"], cmap='Set1')
    plt.title("KNN Predicted Labels (PCA-reduced Data)")
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.colorbar(scatter, label='num_slices')


def plot_3d_clusters(df_pca: pd.DataFrame):
    """
    Plot the test data points and their predicted labels in 3D.

    :param df_pca: A dataframe normalized with PCA.
    """
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection='3d')

    # Plot the test data points, colored by their predicted labels
    scatter = axis.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"], c=df_pca["predicted_label"], cmap='Set1')
    axis.set_title("KNN Predicted Labels (PCA-reduced Data)")
    axis.set_xlabel("Principal Component 1 (PC1)")
    axis.set_ylabel("Principal Component 2 (PC2)")
    axis.set_zlabel("Principal Component 3 (PC3)")
    fig.colorbar(scatter, label='num_slices')
