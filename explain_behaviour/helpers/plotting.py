"""
    Plotting helper functions and color definitions
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import xgboost as xgb

import seaborn as sns
import shap


NICE_COLORS = {'white': 3 * [255],
               'black': 3 * [0],
               'blue': [0, 120, 255],
               'orange': [255, 110, 0],
               'green': [35, 140, 45],
               'red': [200, 30, 15],
               'violet': [220, 70, 220],
               'turquoise': [60, 134, 134],
               'gray': [130, 130, 130],
               'lightgray': 3 * [150],
               'darkgray': 3 * [100],
               'yellow': [255, 215, 0],
               'cyan': [0, 255, 255],
               'dark orange': [244, 111, 22],
               'deep sky blue': [0, 173, 239],
               'deep sky blue dark': [2, 141, 212],
               'tomato': [237, 28, 36],
               'forest green': [38, 171, 73],
               'orange 2': [243, 152, 16],
               'crimson': [238, 34, 53],
               'jaguar': [35, 31, 32],
               'japanese': [59, 126, 52],
               'christi': [135, 208, 67],
               'curious blue': [2, 139, 210],
               'aluminium': [131, 135, 139],
               'buttercup': [224, 146, 47],
               'chateau green': [43, 139, 75],
               'orchid': [125, 43, 139],
               'fiord': [80, 96, 108],
               'punch': [157, 41, 51],
               'lemon': [217, 182, 17],
               'new mpl blue': [31, 119, 180],
               'new mpl red': [214, 39, 40]
               }

for k in NICE_COLORS:
    NICE_COLORS[k] = np.asarray(NICE_COLORS[k])/255.


def set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8, family='Arial'):

    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname(family)

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname(family)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname(family)

    if size_text is not None:
        for at in ax.texts:
            at.set_fontsize(size_text + add_size)
            at.set_fontname(family)



def plot_feature_importance(
        xg_reg,
        color='black',
        one_ferret=False,
        ferrets=None,
        show_plots=False,
        ax=None,
        savefig=False,
        savefig_path=None,
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_importance(xg_reg, ax=ax)
    if one_ferret:
        ax.set_title('feature importances for the XGBoost Correct Release Times model for' + ferrets)
    else:
        ax.set_title('feature importances for the XGBoost Correct Release Times model')
    if show_plots:
        fig.show()

    if savefig:
        fig.savefig(savefig_path, dpi=300)
    return ax

def elbowplot_cumulative_shap(
        shap_values,
        X,
        color='black',
        one_ferret=False,
        ferrets=None,
        show_plots=False,
        ax=None,
        savefig=False,
        savefig_path=None,
    ):
    if ax is None:
        fig, ax = plt.subplots()

    # Ensure shap_values is a 2D array
    if isinstance(shap_values, (int, float)):
        shap_values = np.array([[shap_values]])
    elif len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    feature_importances = np.abs(shap_values).sum(axis=0)
    if len(feature_importances.shape) == 2:
        feature_importances = feature_importances.mean(axis=0)
    sorted_indices = np.argsort(feature_importances)

    sorted_indices = sorted_indices[::-1]
    feature_importances = feature_importances[sorted_indices]
    feature_labels = X.columns[sorted_indices]
    cumulative_importances = np.cumsum(feature_importances)

    # Plot the elbow plot
    ax.plot(feature_labels, cumulative_importances, marker='o', color=color)
    ax.set_xlabel('Features')
    ax.set_ylabel('Cumulative Feature Importance')
    if one_ferret:
        ax.set_title('Elbow Plot of Cumulative Feature Importance \n for the Reaction Time Model \n for ' + ferrets)
    else:
        ax.set_title('Elbow Plot of Cumulative Feature Importance \n for the Reaction Time Model')
    sns.despine(ax=ax, offset=10, trim=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    if savefig:
        fig.tight_layout()
        fig.savefig(savefig_path, dpi=300)
    if show_plots:
        fig.tight_layout()
        fig.show()

    return ax

def shap_summary_plot(
        shap_values,
        X,
        ax=None,
        cmap = "viridis",
        show_plots=False,
        savefig=False,
        savefig_path=None,                
    ):
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    
    # Ensure shap_values matches the shape of X
    if isinstance(shap_values, (int, float)):
        # If scalar, repeat for each row in X
        shap_values = np.tile(shap_values, (X.shape[0], 1))
    elif len(shap_values.shape) == 1:
        # If 1D array, reshape to match X's rows
        shap_values = np.tile(shap_values, (X.shape[0], 1))
    elif len(shap_values.shape) == 2 and shap_values.shape[0] != X.shape[0]:
        # If 2D array but wrong number of rows, reshape
        shap_values = np.tile(shap_values, (X.shape[0] // shap_values.shape[0] + 1, 1))[:X.shape[0]]
    
    plt.sca(ax)
    shap.summary_plot(shap_values, X, show=False, cmap=cmap)
    ax.set_xlabel('SHAP Value (impact on model output)')
    ax.tick_params(axis='y', which='major')
    ax.set_ylabel('Features')
    if savefig:
        fig.savefig(savefig_path, dpi=300)
    if show_plots:
        fig.show()

def plot_permutation_importance(
        result,
        X_test,
        color='black',
        ax=None,
        one_ferret=False,
        ferrets=None,
        show_plots=False,
        savefig=False,
        savefig_path=None,
    ):
    if ax is None:
        fig, ax = plt.subplots()
    sorted_idx = result.importances_mean.argsort()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color = color)
    if one_ferret:
        ax.set_title('Permutation Importances of the \n Reaction Time Model for ' + ferrets, fontsize = 13)
    else:
        ax.set_title('Permutation Importances of the \n Reaction Time Model', fontsize = 13)
    ax.set_xlabel('Permutation Importance')
    sns.despine(ax=ax, offset=10, trim=True)
    if savefig:
        fig.tight_layout()
        fig.savefig(savefig_path, dpi=300)
    if show_plots:
        fig.tight_layout()
        fig.show()
    return ax

def plot_shap_relationship(
        shap_values,
        var_1,
        var_2,
        cmap = "viridis",
        ax=None,
        show_plots=False,
        savefig=False,
        savefig_path=None,
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    shap.plots.scatter(
        shap_values[:, var_1], 
        shap_values[:, var_2], 
        show=False, 
        cmap = cmap, 
        ax=ax,
        )
    
    ax.set_xlabel('SHAP value for ' + str(var_1))
    ax.set_ylabel('SHAP value for ' + str(var_2))
    ax.set_title(f'{var_1} versus {var_2} SHAP values') #, fontsize=18)
    if savefig:
        fig.savefig(savefig_path, dpi=300)
    if show_plots:
        fig.show()
    return ax

def full_shap_plot_reaction_time(
        xg_reg,
        shap_values,
        X,
        X_train,
        X_test,
        perm_result,
        color='black',
        shap_values2=None,
        savefig=False,
        savefig_path=None,
        cmapcustom=mpl.colormaps['viridis'],
    ):
    # mosaic plot
    mosaic_string = '''
                AB
                CB
                '''
    fig, ax_dict = plt.subplot_mosaic(mosaic_string, figsize=(20, 20))

    ax = ax_dict['A']
    ax = elbowplot_cumulative_shap(shap_values, X, ax=ax, color='black')

    ax = ax_dict['B']
    shap_summary_plot(shap_values, X, show_plots=False, ax=ax, cmap=cmapcustom)
    for child in ax.get_children():
        if isinstance(child, mpl.collections.PathCollection):
            child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    ax=ax_dict['C']

    plot_permutation_importance(
            perm_result,
            X_test,
            ax=ax,
    )

    # if shap_values2 is None:
    #     explainer = shap.Explainer(xg_reg)#, X)
    #     shap_values2 = explainer(X_train, check_additivity=False)

    # ax=ax_dict['D']
    # ax = plot_shap_relationship(
    #     shap_values,
    #     'Ferret',
    #     'SNR',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )

    # # colorbar_scatter = fig.axes[1]
    # # colorbar_scatter.set_yticks([-10,120])
    # # colorbar_scatter.set_yticklabels([-10, 'No Noise'], fontsize=18)
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['F1903', 'F2102', 'F2106', 'F2204'], rotation=45)
    # ax.set_ylabel('Influence on reaction time')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'SNR':
    #         cbar_ax = ax
    #         break

    # cbar_ax.set_yticks([0,1,2,3])
    # cbar_ax.set_yticklabels(['-10', '-5', '0', 'No Noise'], rotation=45, fontsize=8)
    # cbar_ax.set_ylabel('SNR', fontsize=8)

    # ax=ax_dict['E']
    # plot_shap_relationship(
    #     shap_values2,
    #     'SNR',
    #     'Target timing',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['-10', '-5', '0', 'No Noise'], rotation=45)
    # ax.set_ylabel('Influence on reaction time')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Target time':
    #         cbar_ax = ax
    #         break
    # colorbar_scatter = fig.axes[1]
    # cbar_ax.set_ylabel('Target Time (s)')
    # yticklabels = cbar_ax.get_yticklabels()
    # cbar_ax.set_yticklabels(yticklabels, fontsize=8)
    

    for k in ax_dict.keys():
        set_font_axes(ax_dict[k], add_size=0, size_ticks=5, size_labels=7,
                      size_text=7, size_title=7, family='Arial')
    for ax in fig.axes:
        if hasattr(ax, 'get_label') and ax.get_label() == '<colorbar>':
            set_font_axes(ax, add_size=0, size_ticks=5, size_labels=7,
                      size_text=7, size_title=7, family='Arial')

    fig.tight_layout()
    return fig, ax_dict

def full_shap_plot_FA_CR(
        xg_reg,
        shap_values,
        X,
        X_train,
        X_test,
        perm_result,
        color='black',
        shap_values2=None,
        savefig=False,
        savefig_path=None,
        cmapcustom=mpl.colormaps['viridis'],
    ):
    # mosaic plot
    mosaic_string = '''
                AB
                CB
                '''
    fig, ax_dict = plt.subplot_mosaic(mosaic_string, figsize=(20, 20))

    ax = ax_dict['A']
    ax = elbowplot_cumulative_shap(shap_values, X, ax=ax, color='black')
    ax.set_title('Elbow Plot of Cumulative Feature Importance \n for the FA vs CR Model', fontsize=7)
    
    ax = ax_dict['B']
    shap_summary_plot(shap_values, X, show_plots=False, ax=ax, cmap=cmapcustom)
    for child in ax.get_children():
        if isinstance(child, mpl.collections.PathCollection):
            child.set_sizes([2.5])  # Replace 'new_size' with the desired size
    ax.set_ylabel('SHAP Value (impact on log(odds) FA')

    ax=ax_dict['C']

    plot_permutation_importance(
            perm_result,
            X_test,
            ax=ax,
    )
    ax.set_title('Permutation Importance for the FA vs CR Model', fontsize=7)

    # if shap_values2 is None:
    #     explainer = shap.Explainer(xg_reg)#, X)
    #     shap_values2 = explainer(X_train, check_additivity=False)

    # ax=ax_dict['D']
    # ax = plot_shap_relationship(
    #     shap_values2,
    #     'Ferret',
    #     'SNR',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )

    # colorbar_scatter = fig.axes[1]
    # colorbar_scatter.set_yticks([-10,120])
    # colorbar_scatter.set_yticklabels([-10, 'No Noise'], fontsize=18)
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['F1903', 'F2102', 'F2106', 'F2204'], rotation=45)
    # ax.set_ylabel('Impact on p(FA)')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'SNR':
    #         cbar_ax = ax
    #         break

    # cbar_ax.set_yticks([0,1,2,3])
    # cbar_ax.set_yticklabels(['-10', '-5', '0', 'No Noise'], rotation=45, fontsize=8)
    # cbar_ax.set_ylabel('SNR', fontsize=8)

    # ax=ax_dict['E']
    # plot_shap_relationship(
    #     shap_values2,
    #     'SNR',
    #     'Trial duration',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['-10', '-5', '0', 'No Noise'], rotation=45)
    # ax.set_ylabel('Impact on p(FA)')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Trial duration':
    #         cbar_ax = ax
    #         break
    # colorbar_scatter = fig.axes[1]
    # cbar_ax.set_ylabel('Trial duration (s)')
    # yticklabels = cbar_ax.get_yticklabels()
    # cbar_ax.set_yticklabels(yticklabels, fontsize=8)
    

    for k in ax_dict.keys():
        set_font_axes(ax_dict[k], add_size=0, size_ticks=5, size_labels=7,
                      size_text=7, size_title=7, family='Arial')
    # for ax in fig.axes:
    #     if hasattr(ax, 'get_label') and ax.get_label() == '<colorbar>':
    #         set_font_axes(ax, add_size=0, size_ticks=5, size_labels=7,
    #                   size_text=7, size_title=7, family='Arial')

    fig.tight_layout()
    return fig, ax_dict

def full_shap_plot(
        xg_reg,
        shap_values,
        X,
        X_train,
        X_test,
        perm_result,
        color='black',
        shap_values2=None,
        savefig=False,
        savefig_path=None,
        cmapcustom=mpl.colormaps['viridis'],
    ):
    # mosaic plot
    mosaic_string = '''
                AB
                CB
                '''
    fig, ax_dict = plt.subplot_mosaic(mosaic_string, figsize=(20, 20))

    ax = ax_dict['A']
    ax = elbowplot_cumulative_shap(shap_values, X, ax=ax, color='black')
    ax.set_title('Elbow Plot of Cumulative Feature Importance', fontsize=7)
    
    ax = ax_dict['B']
    shap_summary_plot(shap_values, X, show_plots=False, ax=ax, cmap=cmapcustom)
    for child in ax.get_children():
        if isinstance(child, mpl.collections.PathCollection):
            child.set_sizes([2.5])  # Replace 'new_size' with the desired size
    ax.set_ylabel('SHAP Value (impact on log(odds)')
    ax=ax_dict['C']

    plot_permutation_importance(
            perm_result,
            X_test,
            ax=ax,
    )
    ax.set_title('Permutation Importance', fontsize=7)

    # if shap_values2 is None:
    #     explainer = shap.Explainer(xg_reg)#, X)
    #     shap_values2 = explainer(X, check_additivity=False)

    # ax=ax_dict['D']
    # ax = plot_shap_relationship(
    #     shap_values2,
    #     'Ferret',
    #     'SNR',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )

    # colorbar_scatter = fig.axes[1]
    # colorbar_scatter.set_yticks([-10,120])
    # colorbar_scatter.set_yticklabels([-10, 'No Noise'], fontsize=18)
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['F1903', 'F2102', 'F2106', 'F2204'], rotation=45)
    # ax.set_ylabel('Impact on p(FA)')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'SNR':
    #         cbar_ax = ax
    #         break

    # cbar_ax.set_yticks([0,1,2,3])
    # cbar_ax.set_yticklabels(['-10', '-5', '0', 'No Noise'], rotation=45, fontsize=8)
    # cbar_ax.set_ylabel('SNR', fontsize=8)

    # ax=ax_dict['E']
    # plot_shap_relationship(
    #     shap_values2,
    #     'SNR',
    #     'Target timing',
    #     ax=ax,
    #     cmap=cmapcustom,
    # )
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(['-10', '-5', '0', 'No Noise'], rotation=45)
    # ax.set_ylabel('Impact on p(Miss)')
    # ax.set_title('Mean SHAP value over ferret ID')
    # for child in ax.get_children():
    #     if isinstance(child, mpl.collections.PathCollection):
    #         child.set_sizes([2.5])  # Replace 'new_size' with the desired size

    # for ax in fig.axes:
    #     if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Target timing':
    #         cbar_ax = ax
    #         break
    # colorbar_scatter = fig.axes[1]
    # cbar_ax.set_ylabel('Target timing (s)')
    # yticklabels = cbar_ax.get_yticklabels()
    # cbar_ax.set_yticklabels(yticklabels, fontsize=8)
    

    for k in ax_dict.keys():
        set_font_axes(ax_dict[k], add_size=0, size_ticks=5, size_labels=7,
                      size_text=7, size_title=7, family='Arial')
    # for ax in fig.axes:
    #     if hasattr(ax, 'get_label') and ax.get_label() == '<colorbar>':
    #         set_font_axes(ax, add_size=0, size_ticks=5, size_labels=7,
    #                   size_text=7, size_title=7, family='Arial')

    fig.tight_layout()
    return fig, ax_dict

def plot_interactions_full(
        model,
        X, 
        shap_values, 
        shap_interaction_values,
        cat_mappings=None,
        interactions = [
                ('Ferret', 'SNR'),
                ('Target timing', 'SNR'),
            ],
        cmap=None,
    ):
    # X_disp to display the correct values
    X_disp = X.copy()

    if cat_mappings is not None:
        for cat, d in cat_mappings.items():
            X_disp[cat] = X_disp[cat].map(d)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15,8),
        # dpi=300,
    )
    
    for (feature_a, feature_b), ax_row in zip(interactions, axes):
        plot_interaction_single(
            X,
            X_disp,
            shap_values, 
            shap_interaction_values, 
            feature_a,
            feature_b,
            ax_row,
            cmap,
            )

    sns.despine(fig, trim=True)

    return fig, axes

def plot_interaction_single(
        X,
        X_disp,
        shap_values,
        shap_interaction_values,
        feature_a,
        feature_b,
        axes,
        cmap,
    ):
    plots_kwargs_scatter = {
        # 'edgecolors' : None,
        'linewidth' : 0.1,
        'alpha' : 0.5,
        'size' : 6,
    }

    plots_kwargs_strip = {
        # 'edgecolors' : None,
        'linewidth' : 0.1,
        'alpha' : 0.5,
        'size' : 4,
    }

    plots_kwargs_point = {
        'linewidth' : 0.1,
        'alpha' : 0.8,
        'markersize' : 10,
    }

    # Obtain the indices of the features to plot
    feature_a_index = X.columns.get_loc(feature_a)
    feature_b_index = X.columns.get_loc(feature_b)

    if X[feature_a].nunique() < 10:
        ax=axes[0]
        sns.stripplot(
            x=X_disp[feature_a],
            y=shap_values[:, feature_a_index],
            hue=X_disp[feature_b],
            palette=cmap,
            ax=ax,
            legend=False,
            dodge=True,
            **plots_kwargs_strip,      
        )
        sns.pointplot(
            x=X_disp[feature_a],
            y=shap_values[:, feature_a_index],
            hue=X_disp[feature_b],
            palette=cmap,
            ax=ax,
            legend=False,
            dodge=True,
            **plots_kwargs_point,      
        )

        ax=axes[1]
        sns.stripplot(
            x=X_disp[feature_a],
            y=shap_interaction_values[:, feature_a_index, feature_a_index],
            ax=ax,
            legend=False,
            **plots_kwargs_strip,      
        )

        ax=axes[2]
        sns.stripplot(
            x=X_disp[feature_a],
            y=shap_interaction_values[:, feature_a_index, feature_b_index],
            hue=X_disp[feature_b],
            palette=cmap,
            legend=False,
            ax=ax,
            **plots_kwargs_strip,      
        )
    else:
        ax = axes[0]
        sns.scatterplot(
            x=X_disp[feature_a],
            y=shap_values[:, feature_a_index],
            hue=X_disp[feature_b],
            palette=cmap,
            ax=ax,
            legend=False,
            **plots_kwargs_scatter,      
        )

        ax=axes[1]
        sns.scatterplot(
            x=X_disp[feature_a],
            y=shap_interaction_values[:, feature_a_index, feature_a_index],
            ax=ax,
            legend=False,
            **plots_kwargs_scatter,      
            )

        ax=axes[2]
        sns.scatterplot(
            x=X_disp[feature_a],
            y=shap_interaction_values[:, feature_a_index, feature_b_index],
            hue=X_disp[feature_b],
            palette=cmap,
            legend=False,
            ax=ax,
            **plots_kwargs_scatter,      
        )

    for ax in [axes[0], axes[2]]:
        # Create a norm object to map the hue data (feature_b values) to the colormap
        norm = mcolors.Normalize(vmin=X_disp[feature_b].min(), vmax=X_disp[feature_b].max())

        # Create a ScalarMappable and initialize with the norm object and colormap
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add the color bar to the axis
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label(feature_b, rotation=270, labelpad=-15)  # Set the label for the color bar
        # Customize the tick labels to only display -10, 0, and replace 10 with 'No noise'
        # cbar.set_ticks([-10, 0, 10])  # Set specific tick positions
        # cbar.set_ticklabels(["-10", "0", "No noise"])  # Set custom tick labels

    axes[0].set_ylabel(f'SHAP value for {feature_a}')
    axes[1].set_ylabel(f'SHAP value for {feature_a} \nwithout the {feature_b} interaction')
    axes[2].set_ylabel(f'SHAP interaction value for \n{feature_a} and {feature_b}')
    
    return axes
