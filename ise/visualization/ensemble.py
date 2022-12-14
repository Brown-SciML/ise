import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import pandas as pd
from ise.utils.data import group_by_run, get_uncertainty_bands

def plot_ensemble(dataset, uncertainty=False, column=None, condition=None, save=None):
    all_trues, all_preds = group_by_run(dataset, column=column, condition=condition)
    mean_true, true_sd, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = get_uncertainty_bands(all_trues,)
    mean_pred, pred_sd, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = get_uncertainty_bands(all_preds,)
    
    true_df = pd.DataFrame(all_trues).transpose()
    pred_df = pd.DataFrame(all_preds).transpose()
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
    axs[0].plot(true_df)
    axs[0].plot(mean_true, 'r-', linewidth=4, label='Mean')
    axs[1].plot(pred_df)
    axs[1].plot(mean_pred, 'r-', linewidth=4, label='Mean')
    if uncertainty and uncertainty.lower() == "confidence":
        axs[0].plot(true_upper_ci, 'b--', linewidth=3, label='5/95% Percentile (True)')
        axs[0].plot(true_lower_ci, 'b--', linewidth=3)
        axs[1].plot(pred_upper_ci, 'b--', linewidth=3, label='5/95% Percentile (Predicted)')
        axs[1].plot(pred_lower_ci, 'b--', linewidth=3)
    
    elif uncertainty and uncertainty.lower() == "quantiles":
        axs[0].plot(pred_upper_q, 'b--', linewidth=3, label='5/95% Confidence (Predicted)')
        axs[0].plot(pred_lower_q, 'b--', linewidth=3)
        axs[1].plot(true_upper_q, 'b--', linewidth=3, label='5/95% Confidence (True)')
        axs[1].plot(true_lower_q, 'b--', linewidth=3)
    
    elif uncertainty and uncertainty.lower() == 'both':
        axs[0].plot(true_upper_ci, 'r--', linewidth=2, label='5/95% Percentile (True)')
        axs[0].plot(true_lower_ci, 'r--', linewidth=2)
        axs[1].plot(pred_upper_ci, 'b--', linewidth=2, label='5/95% Percentile (Predicted)')
        axs[1].plot(pred_lower_ci, 'b--', linewidth=2)
        axs[1].plot(pred_upper_q, 'o--', linewidth=2, label='5/95% Confidence (Predicted)')
        axs[1].plot(pred_lower_q, 'o--', linewidth=2)
        axs[0].plot(true_upper_q, 'k--', linewidth=2, label='5/95% Confidence (True)')
        axs[0].plot(true_lower_q, 'k--', linewidth=2)
        
    
    elif uncertainty and uncertainty.lower() not in ['confidence', 'quantiles']:
        raise AttributeError(f'uncertainty argument must be in [\'confidence\', \'quantiles\'], received {uncertainty}')
    

    axs[0].title.set_text('True')
    axs[0].set_ylabel('True SLE (mm)')
    axs[1].title.set_text('Predicted')
    plt.xlabel('Years since 2015')
    if column is not None and condition is not None:
        plt.suptitle(f'Time Series of ISM Ensemble - where {column} == {condition}')
    else:
        plt.suptitle(f'Time Series of ISM Ensemble')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend()
    
    if save:
        plt.savefig(save)
    
    plt.show()

def plot_ensemble_mean(dataset, uncertainty=False, column=None, condition=None, save=None):
    
    all_trues, all_preds = group_by_run(dataset, column=column, condition=condition)
    mean_true, true_sd, true_upper_ci, true_lower_ci, true_upper_q, true_lower_q = get_uncertainty_bands(all_trues,)
    mean_pred, pred_sd, pred_upper_ci, pred_lower_ci, pred_upper_q, pred_lower_q = get_uncertainty_bands(all_preds,)

    plt.figure(figsize=(15,6))
    plt.plot(mean_true, label='True Mean SLE')
    plt.plot(mean_pred, label='Predicted Mean SLE')
    
    if uncertainty and uncertainty.lower() == "confidence":
        plt.plot(true_upper_ci, 'r--', linewidth=2, label='5/95% Percentile (True)')
        plt.plot(true_lower_ci, 'r--', linewidth=2)
        plt.plot(pred_upper_ci, 'b--', linewidth=2, label='5/95% Percentile (Predicted)')
        plt.plot(pred_lower_ci, 'b--', linewidth=2)
    
    elif uncertainty and uncertainty.lower() == "quantiles":
        plt.plot(pred_upper_q, 'r--', linewidth=2, label='5/95% Confidence (Predicted)')
        plt.plot(pred_lower_q, 'r--', linewidth=2)
        plt.plot(true_upper_q, 'b--', linewidth=2, label='5/95% Confidence (True)')
        plt.plot(true_lower_q, 'b--', linewidth=2)
    
    elif uncertainty and uncertainty.lower() == 'both':
        plt.plot(true_upper_ci, 'r--', linewidth=2, label='5/95% Percentile (True)')
        plt.plot(true_lower_ci, 'r--', linewidth=2)
        plt.plot(pred_upper_ci, 'b--', linewidth=2, label='5/95% Percentile (Predicted)')
        plt.plot(pred_lower_ci, 'b--', linewidth=2)
        plt.plot(pred_upper_q, 'o--', linewidth=2, label='5/95% Confidence (Predicted)')
        plt.plot(pred_lower_q, 'o--', linewidth=2)
        plt.plot(true_upper_q, 'k--', linewidth=2, label='5/95% Confidence (True)')
        plt.plot(true_lower_q, 'k--', linewidth=2)
    
    elif uncertainty and uncertainty.lower() not in ['confidence', 'quantiles']:
        raise AttributeError(f'uncertainty argument must be in [\'confidence\', \'quantiles\'], received {uncertainty}')
    
    else:
        pass
    
    if column is not None and condition is not None:
        plt.suptitle(f'ISM Ensemble Mean SLE over Time - where {column} == {condition}')
    else:
        plt.suptitle(f'ISM Ensemble Mean over Time')
    plt.xlabel('Years since 2015')
    plt.ylabel('Mean SLE (mm)')
    plt.legend()
    
    if save:
        plt.savefig(save)
    plt.show()
    
    
# def generate_ts_comparison(dataset,  plot=True, bands=False):
    
    

#     if plot:
#         fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)

#         for i, array in enumerate(all_trues):
#             axs[0].plot(array)
#         axs[0].plot(mean_true, 'r-', linewidth=4, label='Mean')
#         if bands:
#             axs[0].plot(true_quantiles[0,:], 'r--', linewidth=4, label='5/95% Percentile')
#             axs[0].plot(true_quantiles[1,:], 'r--', linewidth=4)
#         axs[0].title.set_text('True')
#         axs[0].set_ylabel('True SLE (mm)')


#         for i, array in enumerate(all_preds):
#             axs[1].plot(array)
#         axs[1].plot(mean_pred, 'r-', linewidth=4, label='Mean')
#         if bands:
#             axs[1].plot(pred_quantiles[0,:], 'r--', linewidth=4, label='5/95% Percentile')
#             axs[1].plot(pred_quantiles[1,:], 'r--', linewidth=4)
#         axs[1].title.set_text('Predicted')
#         plt.xlabel('Years since 2015')
#         if column is not None and condition is not None:
#             plt.suptitle(f'Time Series of ISM Ensemble - where {column} == {condition}')
#         else:
#             plt.suptitle(f'Time Series of ISM Ensemble')
#         plt.subplots_adjust(wspace=0, hspace=0)
#         plt.show()

#         print('')

#         plt.figure(figsize=(15,6))
#         plt.plot(mean_true, label='True Mean SLE')
#         plt.plot(mean_pred, label='Predicted Mean SLE')
#         if bands:
#             plt.plot(true_quantiles[0,:], 'r--', linewidth=2, label='5/95% Percentile (True)')
#             plt.plot(true_quantiles[1,:], 'r--', linewidth=2)
#             plt.plot(pred_quantiles[0,:], 'b--', linewidth=2, label='5/95% Percentile (Predicted)')
#             plt.plot(pred_quantiles[1,:], 'b--', linewidth=2)
#         if column is not None and condition is not None:
#             plt.suptitle(f'ISM Ensemble Mean SLE over Time - where {column} == {condition}')
#         else:
#             plt.suptitle(f'ISM Ensemble Mean over Time')
#         plt.xlabel('Years since 2015')
#         plt.ylabel('Mean SLE (mm)')
#         plt.legend()
#         plt.show()
#     return all_trues, mean_true, all_preds, mean_pred