import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from plotly.offline import iplot
from plotly import graph_objs as go
from plotly import subplots as psubplots
from plotly import colors as pcolors

def set_size(width, fraction=1):
    """
    Returns optimal figure dimension for a latex
    plot, depending on the requested width.
    
    Args:
        width (int)     : Width of the figure
        fraction (float): Fraction of the width
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return (fig_width_in, fig_height_in)


def apply_latex_style():
    """
    Sets the necessary matplotlib and seaborn parameters
    to draw a plot using latex.
    """
    sns.set(rc={'figure.figsize':set_size(250),
                'text.usetex':True,
                'font.family': 'serif',
                'axes.labelsize': 8,
                'font.size': 8,
                'legend.fontsize': 8,
                'legend.labelspacing': 0.25,
                'legend.columnspacing': 0.25,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,}, context='paper')
    sns.set_style('white')
    sns.set_style(rc={'axes.grid':True, 'font.family': 'serif'})
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath,bm}"]


def find_key(df, key_list, separator=':'):
    """
    Checks if a DataFrame contains any of the keys listed
    in a character-separated string.
 
    Args:
        df (pandas.DataFrame): Pandas dataframe (or dictionary) containing data
        key_lit (str)        : Character-separated list of keys
        separator (str)      : Separation character between keys
    Returns:
        str: Key found
        str: Name of the first key (for legend purposes)
    """
    key_list  = key_list.split(separator)
    key_name  = key_list[0]
    key_found = np.array([k in df.keys() for k in key_list])
    if not np.any(key_found):
        raise KeyError('Could not find the keys provided:', key_list)
    key = key_list[np.where(key_found)[0][0]]
    return key, key_name


def get_training_df(log_dir, keys, prefix='train'):
    """
    Finds all training log files inside the specified directory
    and concatenates them. If the range of iterations overlap, keep only
    that from the file started further in the training.
 
    Assumes that the formatting of the log file names is of the form
    `prefix-x.csv`, with `x` the number of iterations.
 
    Args:
        log_dir (str): Path to the directory that contains the training log files
        keys (list)  : List of quantities of interest
        prefix (str) : Prefix shared between training file names (default: `train`)
    Returns:
        pandas.DataFrame: Combined training log data
    """
    log_files  = np.array(glob.glob(f'{log_dir}/{prefix}*'))
    end_points = np.array([int(f.split('-')[-1].split('.csv')[0]) for f in log_files])
    order      = np.argsort(end_points)
    end_points = np.append(end_points[order], 1e12)
    log_dfs    = []
    for i, f in enumerate(log_files[order]):
        df = pd.read_csv(f, nrows=end_points[i+1]-end_points[i])
        for key_list in keys:
            key, key_name = find_key(df, key_list)
            df[key_name] = df[key]
        log_dfs.append(df)

    if not len(log_dfs):
        raise FileNotFoundError(f'Found no train log with prefix \'{prefix}\' under {log_dir}')

    return pd.concat(log_dfs, sort=True) 


def get_validation_df(log_dir, keys, prefix='inference'):
    """
    Finds all validation log files inside the specified directory
    and build a single dataframe out of them. It returns the mean and
    std of the requested keys for each file.
 
    Assumes that the formatting of the log file names is of the form
    `prefix-x.csv`, with `x` the number of iterations.
    
    The key list allows for `:`-separated names, in case separate files
    use different names for the same quantity.

    Args:
        log_dir (str): Path to the directory that contains the validation log files
        keys (list)  : List of quantities to get mean/std for
        prefix (str) : Prefix shared between validation file names (default: `inference`)
    Returns:
        pandas.DataFrame: Combined validation log data
    """
    # Initialize a dictionary
    val_data = {'iter':[]}
    for key in keys:
        key_name = key.split(':')[0]
        val_data[key_name+'_mean'] = []
        val_data[key_name+'_err'] = []

    # Loop over validation log files
    log_files = np.array(glob.glob(f'{log_dir}/{prefix}*'))
    for log_file in log_files:
        df = pd.read_csv(log_file)
        it = int(log_file.split('/')[-1].split('-')[-1].split('.')[0])
        val_data['iter'].append(it-1)
        for key_list in keys:
            key, key_name = find_key(df, key_list)
            val_data[f'{key_name}_mean'].append(df[key].mean())
            val_data[f'{key_name}_err'].append(df[key].std()/np.sqrt(len(df[key])))
 
    args = np.argsort(val_data['iter'])
    for key, val in val_data.items():
        val_data[key] = np.array(val)[args]
 
    return pd.DataFrame(val_data)


def draw_training_curves(log_dir, models, metrics,
                         limits={}, model_names={}, metric_names={},
                         max_iter=-1, step=1, smoothing=1, iter_per_epoch=-1,
                         print_min=False, print_max=False,
                         interactive=True, same_plot=True, paper=False, leg_ncols=1,
                         figure_name='', train_prefix='train', val_prefix='inference'):
    """
    Finds all training and validation log files inside the specified 
    directory and draws an evolution plot of the request quantities.

    Args:
        log_dir (str)       : Path to the directory that contains the folder with log files
        models (list)       : List of model (folder) names under the main directory
        metrics (list)      : List of quantities to draw
        limits (list/dict)  : List of y boundaries for the plot (or dictionary of y boundaries, one per metric)
        model_names (dict)  : Dictionary which maps raw model names to model labels (default: `{}`)
        metric_names (dict) : Dictionary which maps raw metric names to metric labels (default: `{}`)
        max_iter (int)      : Maximum number of interation to include in the plot (default: `-1`)
        step (int)          : Step between two successive iterations that are represented (default: `1`)
        smoothing (int)     : Number of iteration over which to average the metric value (default: `1`)
        iter_per_epoch (float): Number of iterations to complete an epoch (default: `-1`, figures it out from train log)
        interactive (bool)  : Use plotly to draw (default: `True`)
        same_plot (bool)    : Draw all model/metric pairs on a single plot (default: `True`)
        paper (bool)        : Format plot for paper, using latex (default: `False`)
        leg_ncols (int)     : Number of columns in the legend (default: `1`)
        figure_name (str)   : Name of the figure. If specified, figure is saved (default: `''`)
        train_prefix (str)  : Prefix shared between training file names (default: `train`)
        val_prefix (str)    : Prefix shared between validation file names (default: `inference`)
    """
    # Set the style
    plotly_colors = pcolors.convert_colors_to_same_type(pcolors.DEFAULT_PLOTLY_COLORS, 'tuple')[0]
    if not interactive:
        cr_char = '\n'
        if paper:
            apply_latex_style()
            linewidth  = 0.5
            markersize = 1
        else:
            sns.set(rc={'figure.figsize':(16,9)}, context='notebook', font_scale=2)
            sns.set_style('white')
            sns.set_style(rc={'axes.grid':True})
            linewidth  = 2
            markersize = 10
    else:
        graphs = []
        cr_char = '<br>'
        converter = lambda color: 'rgba({}, {}, {}, 0.5)'.format(*color)
        plotly_colors = pcolors.color_parser(plotly_colors, converter)
        layout = go.Layout(template='plotly_white', width=1000, height=500, margin=dict(r=20, l=20, b=20, t=20), 
                           xaxis=dict(title=dict(text='Epochs', font=dict(size=20)), tickfont=dict(size=20), linecolor='black', mirror=True),
                           yaxis=dict(title=dict(text='Metric', font=dict(size=20)), tickfont=dict(size=20), linecolor='black', mirror=True),
                           legend=dict(font=dict(size=20), tracegroupgap=1))
        if len(models) == 1 and same_plot:
            layout['legend']['title'] = model_names[models[0]] if models[0] in model_names else models[0]
 
    # If there is >1 subplot, prepare the canvas
    if not same_plot:
        if not interactive:
            fig, axes = plt.subplots(len(metrics), sharex=True)
            fig.subplots_adjust(hspace=0)
            for axis in axes:
                axis.set_facecolor('white')
        else:
            fig = psubplots.make_subplots(rows=len(metrics), shared_xaxes=True, vertical_spacing=0)
            for i in range(len(metrics)):
                if i > 0:
                    layout[f'xaxis{i+1}'] = layout['xaxis']
                    layout[f'yaxis{i+1}'] = layout['yaxis']
                layout[f'xaxis{i+1}']['title']['text'] = '' if i < len(metrics)-1 else 'Epochs'
                layout[f'yaxis{i+1}']['title']['text'] = metric_names[metrics[i]] if metrics[i] in metric_names else metrics[i]
                if metrics[i] in limits and len(limits[metrics[i]]) == 2:
                    layout[f'yaxis{i+1}']['range'] = limits[metrics[i]]

            fig.update_layout(layout)
    elif interactive:
        if isinstance(limits, list) and len(limits) == 2:
            layout['yaxis']['range'] = limits
        if len(metrics) == 1:
            layout['yaxis']['title']['text'] = metric_names[metrics[0]] if metrics[0] in metric_names else metrics[0]
        fig = go.Figure(layout=layout)

    # Get the DataFrames for the requested models/metrics
    dfs, val_dfs, colors = {}, {}, {}
    for i, key in enumerate(models):
        log_subdir = log_dir+key
        dfs[key] = get_training_df(log_subdir, metrics, train_prefix)
        val_dfs[key] = get_validation_df(log_subdir, metrics, val_prefix)
        colors[key] = plotly_colors[i]

    # Loop over the requested metrics
    for i, metric_list in enumerate(metrics):
        # Get a graph per training campaign
        for j, key in enumerate(dfs.keys()):
            # Get the necessary data
            epoch_train  = dfs[key]['epoch'][:max_iter:step]
            metric, metric_name = find_key(dfs[key], metric_list)
            metric_train = dfs[key][metric][:max_iter:step] if smoothing < 2 else dfs[key][metric][:max_iter].rolling(smoothing, min_periods=1, center=True).mean()[::step]
            draw_val     = bool(len(val_dfs[key]['iter']))
            if draw_val:
                mask_val    = val_dfs[key]['iter'] < max_iter if max_iter > -1 else val_dfs[key]['iter'] < 1e12
                iter_val    = val_dfs[key]['iter'][mask_val]
                if iter_per_epoch < 0:
                    epoch_val   = [dfs[key]['epoch'][dfs[key]['iter'] == it] for it in iter_val]
                    epoch_val   = np.array([float(e) if len(e)==1 else -1 for e in epoch_val])
                    mask_val   &= epoch_val > -1
                    iter_val    = iter_val[epoch_val > -1]
                    epoch_val   = epoch_val[epoch_val > -1]
                else:
                    epoch_val = iter_val/iter_per_epoch
                metricm_val = val_dfs[key][metric_name+'_mean'][mask_val]
                metrice_val = val_dfs[key][metric_name+'_err'][mask_val]
 
            # Pick a label for this specific model/metric pair
            if not same_plot:
                label = model_names[key] if key in model_names else key
            else:
                if len(models) == 1:
                    label = metric_names[metric_name] if metric_name in metric_names else metric_name
                elif len(metrics) == 1:
                    label = model_names[key] if key in model_names else key
                else:
                    label = f'{metric_names[metric_name] if metric_name in metric_names else metric_name} ({model_names[key] if key in model_names else key})'
                if print_min and draw_val:
                    label += f'{cr_char}Min: {iter_val[np.argmin(metricm_val)]:d}'
                if print_max and draw_val:
                    label += f'{cr_char}Max: {iter_val[np.argmax(metricm_val)]:d}'

            # Prepare the relevant plots
            color = colors[key] if not same_plot else plotly_colors[i*len(models)+j]
            if not interactive:
                axis = plt if same_plot else axes[i]
                axis.plot(epoch_train, metric_train, label=label, color=color, alpha=0.5, linewidth=linewidth)
                if draw_val:
                    axis.errorbar(epoch_val, metricm_val, yerr=metrice_val, fmt='.', color=color, linewidth=linewidth, markersize=markersize)
            else:
                legendgroup = f'group{i*len(models)+j}'
                graphs += [go.Scatter(x=epoch_train, y=metric_train, name=label, line=dict(color=color), legendgroup=legendgroup, showlegend=(same_plot | (not same_plot and not i)))]
                if draw_val:
                    hovertext = [f'(Iteration: {iter_val[i]:d})' for i in range(len(iter_val))]
                    # hovertext = [f'(Iteration: {iter_val[i]:d}, Epoch: {epoch_val[i]:0.3f}, Metric: {metricm_val[i]:0.3f})' for i in range(len(iter_val))]
                    graphs += [go.Scatter(x=epoch_val, y=metricm_val, error_y_array=metrice_val, mode='markers', hovertext=hovertext, marker=dict(color=color), legendgroup=legendgroup, showlegend=False)]

    if not interactive:
        if not same_plot:
            for i, metric in enumerate(metrics):
                metric_name = metric.split(':')[0]
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric_names[metric_name] if metric_name in metric_names else metric_name)
                if metric_name in limits and len(limits[metric_name]) == 2:
                    axes[i].set_ylim(limits[metric_name])
            axes[0].legend(ncol=leg_ncols)
        else:
            plt.xlabel('Epochs')
            ylabel = metric_names[metrics[0]] if metrics[0] in metric_names else metrics[0]
            plt.ylabel(ylabel if len(metrics) == 1 else 'Metric')
            plt.gca().set_ylim(limits)
            legend_title = model_names[models[0]] if models[0] in model_names else models[0]
            plt.legend(ncol=leg_ncols, title=legend_title if len(models)==1 else None)
        if len(figure_name):
            plt.savefig(f'{figure_name}.png', bbox_inches='tight')
            plt.savefig(f'{figure_name}.pdf', bbox_inches='tight')
        plt.show()
    else:
        if not same_plot:
            rows = list(np.arange(len(metrics), step=1./(2**draw_val*len(models))).astype(int)+1)
            cols = list(np.ones(2**draw_val*len(models)*len(metrics), dtype=int))
            fig.add_traces(graphs, rows=rows, cols=cols)
        else:
            fig.add_traces(graphs)
        iplot(fig)
