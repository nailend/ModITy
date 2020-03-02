# utf-8

#### Bachelor Thesis: Julian Endres
## 2019.12.13
# FG: Energie- und Ressourcenmanagement, Tu Berlin

# ### ModITy - Model for the Identification of Typregions

import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

from matplotlib.colors import to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from IPython.display import display
import logging

# ### ModITy - Model for the Identification of Typregions


class Project:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.wrap = True
        self.path = os.getcwd()
        self.make_directory(self.input_folder)
        self.make_directory('graphics')
        self.make_directory('output')

        self.input_path = self.get_input_path()
        self.input_files = None
        self.df_transfer = None
        self.detect_csvs()

        self.all_columns = None
        self.all_index = None
        self.df_raw = None
        self.df_data = None
        self.memory = None
        self.cmap_col = "Spectral_r"

    def jupyter_wrapper(self, dataframe):
        """ a wrapper function, test if still needed after import of
        IPython.display """
        if self.wrap:
            display(dataframe)
        else:
            print(dataframe)

    def get_input_path(self):
        """joins path and name of input folder"""
        input_path = self.path + '/' + self.input_folder
        print('----------------------------------------------------------')
        print("Path of Input folder saved in instance 'input_path'.")
        return input_path

    @staticmethod
    def str_date_time():
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y%m%d_%H-%M-%S")  # str format
        return date_time

    @staticmethod
    def make_directory(folder_name, **kwargs):
        """creates a folder if not yet existing
        :parameter
        folder_name: str, Name of the folder to be created

        """
        subfolder = kwargs.get('subfolder', False)

        existing_folders = next(os.walk('.'))[1]
        if folder_name in existing_folders:
            print('----------------------------------------------------------')
            print('Folder "' + folder_name + '" already exists in current directory.')
        else:
            path = "./" + folder_name
            os.mkdir(path)
            print('----------------------------------------------------------')
            print('Created folder "' + folder_name + '" in current directory.')
        if subfolder:
            if isinstance(subfolder, str):
                existing_folders = next(os.walk('./' + folder_name))[1]
                if subfolder in existing_folders:
                    print('----------------------------------------------------------')
                    print('Subfolder "' + subfolder + '" already exists in ./' + folder_name + '.')
                else:
                    path = "./" + folder_name + "/" + subfolder
                    os.mkdir(path)
                    print('----------------------------------------------------------')
                    print('Created subfolder "' + subfolder + '" in ./' + folder_name + '.')
            elif ~isinstance(subfolder, bool):
                print('The name of the subfolder has to be of type str')

    def read_transfertable(self, path):
        """reads transfertabelle from input_files
        :parameter
        path of the table
        """
        try:
            df_transfer = pd.read_csv(path , sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df_transfer = pd.read_csv(self.input_path + '/' + input_files[
                'transfer'][0], sep=';', encoding='latin1')

        df_transfer['AGS'] = df_transfer['AGS'].astype(str).apply(
            lambda x: x.zfill(5))
        df_transfer['PLZ'] = df_transfer['PLZ'].astype(str).apply(
            lambda x: x.zfill(5))
        df_transfer.set_index('Index', drop=True, inplace=True)
        return df_transfer

    def detect_csvs(self):
        """ looks for all csv files in input_path
        sort the files by beginning
        NUTS3, AGS, PLZ
        reads "transfertabelle" and save it as an attribute
        """

        path = self.input_path

        # initiate
        files_nuts3 = []
        files_ags = []
        files_plz = []
        files_transfer = []

        # scan folder
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    if file.startswith('NUTS3'):
                        files_nuts3.append(file)
                    elif file.startswith('AGS'):
                        files_ags.append(file)
                    elif file.startswith('PLZ'):
                        files_plz.append(file)
                    elif file.startswith('Transfer'):
                        files_transfer.append(file)

        input_files = {'NUTS3': files_nuts3, 'AGS': files_ags, 'PLZ': files_plz,
                       'transfer': files_transfer}
        df_input_files = pd.DataFrame({key: pd.Series(value) for key, value in input_files.items()})

        if len(files_transfer) == 1:
            path = self.input_path + '/' + files_transfer[0]
            df_transfer = self.read_transfertable(path)
            self.df_transfer = df_transfer
        else:
            print('None or multiple "Transfertabelle" were found.')
            print('Import Transfertabelle manually with with pandas and '
                  'pass to import_transfer_file(file)')
        self.input_files = df_input_files
        self.jupyter_wrapper(df_input_files)

    def assign_nuts3(self, code_type):
        """ NUTS3 codes are assigned if correct code_type of dataset is passed.
        a few checks are done, if dataset are complete
        :parameter
        code_type: str, AGS, PLZ, NUTS3
        """
        df_transfer = self.df_transfer.copy()
        df_data = self.df_raw.copy()

        possible_code_types = ['AGS', 'PLZ', 'NUTS3']
        if code_type not in possible_code_types:
            raise ValueError('The code_type memory_set must be one of the following set: ' \
                             '"{}"'.format('","'.join(possible_code_types)))
        if code_type not in set(df_data.columns):
            raise ValueError('Column of {} is missing.'.format(code_type))

        if 'NUTS3' in df_data.columns:
            if set(df_data['NUTS3']) == set(df_transfer['NUTS3']):
                df_data.set_index('NUTS3', drop=True, inplace=True)
                print('NUTS3 assignment detected! Index was set!')
            else:
                df_data.set_index('NUTS3', drop=True, inplace=True)
                print('NUTS3 assignment detected! Data Set is not complete!')

        else:
            # convert column to type string
            df_data[code_type] = df_data[code_type].astype(str)
            df_transfer[code_type] = df_transfer[code_type].astype(str)

            # extend strings to "digits" length by adding zeros in the front
            if code_type == 'AGS':
                digits = 5
            elif code_type == 'PLZ':
                digits = 5

            df_data[code_type] = df_data[code_type].apply(
                lambda x: x.zfill(digits))
            df_transfer[code_type] = df_transfer[code_type].apply(
                lambda x: x.zfill(digits))

            # compare code_type sets
            if set(df_data[code_type]) != set(df_transfer[code_type]):
                missing = set(df_data[code_type]).difference(set(df_transfer[code_type]))
                print('The following ' + str(code_type) + ' is/are missing!')
                self.jupyter_wrapper(missing)
                raise ValueError('There is a difference in the regional codes!')

            df_data.set_index([code_type], inplace=True)
            # merge NUTS3
            df_data = pd.merge(df_data, df_transfer[[code_type, 'NUTS3']],
                               how='inner', left_index=True, right_on=code_type)
            df_data.set_index('NUTS3', drop=True, inplace=True)
            df_data.drop(columns=code_type, inplace=True)
            print('NUTS3 assigned data saved to df_data.')

        self.df_data = df_data
        self.all_columns = df_data.columns
        self.all_index = df_data.index

    def import_transfer_file(self, file):
        """manually import transfer table, import table with pandas as
        pd.DataFrame and pass to function
        :parameter
        file: pd.DataFrame, transfer table as pd.DataFrame
        """
        df_transfer = file
        df_transfer['AGS'] = df_transfer['AGS'].astype(str).apply(
            lambda x: x.zfill(5))
        df_transfer['PLZ'] = df_transfer['PLZ'].astype(str).apply(
            lambda x: x.zfill(5))
        df_transfer['index'] = df_transfer['NUTS3']
        df_transfer.set_index('Index', drop=True, inplace=True)
        self.df_transfer = df_transfer

    def import_data_file(self, file, code_type):
        self.df_raw = file.copy()
        print('file imported')
        self.assign_nuts3(code_type)
        self.initiate_memory()

    @staticmethod
    def normalization(df_data, scaler):
        """Transforms every feature of the input DataFrame
         :parameter
         df_data = pd.DataFrame, input Matrix
         scaler = 'standard', 'minmax', "robust",

         :return
         df_transform = pd.DataFrame, transformed Matrix
        """
        df_transform = df_data.copy()
        possible_scaler = ["standard", "minmax", "robust"]
        if scaler not in possible_scaler:
            raise ValueError('The scaler must be one of the following set: '
                             '"{}"'.format('","'.join(possible_scaler)))

        if scaler == "standard":
            # math: z = (x - u) / s
            # u = mean, s = standard
            df_transform.loc[:, :] = scale(df_transform,
                                           axis=0)

        elif scaler == "minmax":
            # scaling each feature to a given range min,max (0,1)
            df_transform.loc[:, :] = minmax_scale(df_transform,
                                                  feature_range=(0, 1),
                                                  axis=0)

        elif scaler == "robust":
            # Interquartile range scaling (0,1)
            df_transform.loc[:, :] = robust_scale(df_transform,
                                                  axis=0)

        return df_transform

    def initiate_memory(self):
        """initiate the memory """
        df_transfer = self.df_transfer.copy()
        df_data = self.df_data.copy()
        memory = {0: {'samples': dict.fromkeys(df_transfer.index, True),
                      'features': dict.fromkeys(df_data.columns, True),
                      'outlier': (None, None, None),
                      'scores': None,
                      'labels': {},
                      'scaler': None,
                      'borderliner': {},
                      }}

        self.memory = memory
        print('Memory was initiated!')

    def outliers(self, memory_set=0, scaler='robust', save=False):
        """ Detect outliers with IQR method exceeding specific thresholds.
        Thresholds are handed via input()
        figure with threshold will be save if save=True

        :parameter
        memory_set: int, specific memory_set wich shall be used
        scaler: str, transfomer to be used, default='robust
        save: bool, saves figure
        """
        df_data = self.df_data.copy()
        df_transfer = self.df_transfer.copy()
        memory = self.memory

        if memory_set not in range(max(memory.keys()) + 1):
            raise ValueError('The key does not yet exist in the memory.')

        samples = {key for key, value in memory[memory_set]['samples'].items()
                   if value}
        features = {key for key, value in memory[memory_set]['features'].items()
                    if value}
        df_data = df_data.loc[samples, features].copy()

        dict_outliers = {}
        new_key = max(memory.keys()) + 1

        # take not selected data
        df_robust = self.normalization(df_data, scaler)
        ax = df_robust.plot(figsize=(10, 10))
        plt.show()

        print('set positive threshold or enter "None":')
        pos_threshold = input()
        print('-----------------------')
        print('set negative threshold or enter "None":')
        neg_threshold = input()

        if pos_threshold == 'None':
            pos_threshold = None
            # pos_outliers = None
            df_pos_features = pd.DataFrame()
            df_pos_samples = pd.DataFrame()
        else:
            pos_threshold = float(pos_threshold)
            df_pos_samples = (df_robust > pos_threshold).sum(
                axis=1).sort_values(ascending=False).to_frame(
                'exceeded pos. limits')
            # dict_outliers.update(dict.fromkeys(pos_samples.index, False))
            # oop einbinden self.df_transfer
            df_pos_features = (df_robust > pos_threshold).sum().sort_values(
                ascending=False).to_frame('exceeded pos. limits')

        if neg_threshold == 'None':
            neg_threshold = None
            # neg_outliers = None
            df_neg_features = pd.DataFrame()
            df_neg_samples = pd.DataFrame()
        else:
            neg_threshold = float(neg_threshold)
            df_neg_samples = (df_robust < neg_threshold).sum(
                axis=1).sort_values(ascending=False).to_frame(
                'exceeded neg. limits')
            # dict_outliers.update(dict.fromkeys(neg_samples.index, False))
            # oop einbinden self.df_transfer
            df_neg_features = (df_robust < neg_threshold).sum().sort_values(
                ascending=False).to_frame('exceeded neg. limits')

        print('----------------------------------')
        print('Features exceeding positiv threshold for multiple samples')
        df_features = df_pos_features.join(df_neg_features)
        features = df_features.sum(axis=1) > 0
        self.jupyter_wrapper(df_features[features])

        print('----------------------------------')
        print('Samples exceeding positiv threshold for multiple features')
        df_samples_out = df_pos_samples.join(df_neg_samples)
        samples_out = df_samples_out.sum(axis=1) > 0
        self.jupyter_wrapper(df_samples_out.join(df_transfer['Name'])[samples_out])
        dict_outliers.update(
            dict.fromkeys(samples_out.index[samples_out], False))

        print('----------------------------------')
        print('----------------------------------')
        print('Do you want to drop these samples? (y/n)')
        confirmation = input()
        if confirmation == "y":

            # new dict_entry = copy of initialisation
            new_sample_memory_set = deepcopy(memory[memory_set])
            new_sample_memory_set['samples'].update(dict_outliers)
            new_sample_memory_set['outlier'] = (memory_set, pos_threshold,
                                             neg_threshold)
            self.memory.update({new_key: new_sample_memory_set})
            print('----------------------------------')
            print('manipulated and saved as memory_set={}'.format(new_key))

            if save:
                # ax = fig.gca()
                ax.hlines(y=pos_threshold,
                          xmin=0, xmax=len(df_robust),
                          colors='red')
                ax.hlines(y=neg_threshold,
                          xmin=0, xmax=len(df_robust),
                          colors='red')
                fig = ax.get_figure()
                now = self.str_date_time()
                figname = "./graphics/" + now + '_Outliers_memory_set_' + str(
                    memory_set) + ".pdf"
                fig.savefig(figname, bbox_inches='tight')
                # self.logger.info('Figure saved at {}'.format(figname))
                print('Outliers saved as ' + figname)

        else:
            print('----------------------------------')
            print('The process was canceled!')

    def violinplots(self, memory_set, scaler='robust', save=False):
        """ plots violin plots for every feature in the memory set
        :parameter
        memory_set: int, specific memory_set wich shall be used
        scaler: str, transfomer to be used, default='robust
        save: bool, saves figure

        """
        df_data = self.df_data.copy()
        memory = self.memory

        samples = {key for key, value in memory[memory_set]['samples'].items()
                   if value}
        features = {key for key, value in
                    memory[memory_set]['features'].items() if value}
        df_data = df_data.loc[samples, features].copy()

        df_transform = self.normalization(df_data, scaler)

        num_feat = len(df_transform.columns)
        fig, ax = plt.subplots(figsize=(10, num_feat *2))

        # define colors
        cmap = sns.color_palette("Spectral", num_feat)
        # Plot the orbital period with horizontal boxes
        sns.violinplot(orient='h', data=df_transform, scale='count',
                       inner='quartile',
                       whis="range", palette=cmap, ax=ax)

        # Add in points to show each observation
        sns.swarmplot(orient='h', data=df_transform,
                      size=2, color=".1", linewidth=0, ax=ax)

        ax.xaxis.grid(True)
        ax.set(ylabel="")
        fig.suptitle('Violinplots of {}-transformed memory set {}'.format(
            scaler, memory_set), fontsize=14)#, fontweight='bold')
        plt.subplots_adjust(left=0.25)
        plt.show()
        if save:
            now = self.str_date_time()
            figname = './graphics/{}_Violinplots_memory_set_{}.pdf'.format(
                now, memory_set)
            fig.savefig(figname, bbox_inches='tight')
            print('Violinplots saved as ' + figname)

    @staticmethod
    def highlight_samples(data, color1='green', color2='red', **kwargs):
        """
        highlight kwargs in a Series or DataFrame
        """
        highlight = kwargs.get('highlight', False)
        color1 = 'background-color: {}'.format(color1)
        color2 = 'background-color: {}'.format(color2)
        return [color1 if v else color2 for v in highlight[data.name]]

    # show all features
    def show_samples(self):
        """creates a table of all excluded samples in the memory_sets:
        doesnt work after select_samples as dictionary is extended and
        sorted. different highlighting method needs to be developed"""
        df_transfer = self.df_transfer.copy()
        memory = self.memory

        # idenify all trues in memory[i]['samples']
        outlier_set = []
        for i in list(memory):
            outlier_set += [item for item, value in memory[i][
                'samples'].items() if not value]
        outlier_set = set(outlier_set)

        # generate Dataframe for highlighting
        highlight = []
        for i in list(memory):
            highlight.append(
                [value for item, value in memory[i]['samples'].items() if item in outlier_set])
        highlight = list(zip(*highlight))
        highlight = pd.DataFrame(highlight,
                                 index=list(outlier_set),
                                 columns=list(memory))

        # generate dataframe with equal samples for each memory_set
        frame = [list(outlier_set)] * len(memory)
        frame = list(zip(*frame))
        df_samples = pd.DataFrame(frame,
                                  index=df_transfer['Name'][outlier_set],
                                  columns=list(memory))
        df_samples.index.name = 'samples'
        df_samples.columns.name = 'memory set'
        print('red = dropped sample')
        print('green =  included sample')
        self.jupyter_wrapper(df_samples.style.apply(self.highlight_samples, highlight=highlight))

    def correlation(self, method='pearson', memory_set=None, heatmap=False,
                    threshold=None, save=False):
        """computes the correlation matrix and can show the results in a heatmap.

        :parameter
        df_data : input Dataframe; columns = features; index = samples
        method : {'pearson', 'kendall', 'spearman'}
        memory_set : int, default = None, which takes the latest
        heatmap : boolean, default = False
        threshold : float, default= None
        save : boolean, default = False

        :return
        list of features > threshold
        """
        df_data = self.df_data.copy()
        memory = self.memory
        if memory_set is None:
            memory_set = max(memory.keys())
        # self.logger.info('Selected memory set {}'.format(memory_set))

        if memory_set not in range(max(memory.keys()) + 1):
            raise ValueError('The key does not yet exist in the memory.')

        samples = {key for key, value in memory[memory_set]['samples'].items()
                   if value}
        features = {key for key, value in memory[memory_set]['features'].items()
                    if value}
        df_data = df_data.loc[samples, features].copy()

        if not isinstance(heatmap, bool):
            raise ValueError('Input value for heatmap has to be of type bool')

        possible_methods = ['pearson', 'kendall', 'spearman']
        if method not in possible_methods:
            raise ValueError('The method must be one of the following set: '
                             '"{}"'.format('","'.join(possible_methods)))

        df_corr = df_data.corr(method=method)

        if heatmap:
            # only lower left matrix
            #mask = np.zeros_like(df_corr)
            #mask[np.triu_indices_from(mask)] = True
            # without diagonal values
            mask = np.identity(df_corr.shape[0])
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.suptitle('Heatmap: {} correlation of memory_set {}'.format(
                method, memory_set),
                         fontsize=20)
            cmap = cm.get_cmap(self.cmap_col)
            with sns.axes_style("white"):
                sns.heatmap(df_corr, mask=mask, square=True, linewidths=1,
                            vmax=1, vmin=-1, cbar=False, fmt='.2f',
                            annot=True, cmap=cmap, ax=ax)#"RdBu_r")
                # bugfix matllotlib<3.0.3 truncated heatmap
                b, t = ax.get_ylim()  # discover the values for bottom and top
                b += 0.5  # Add 0.5 to the bottom
                t -= 0.5  # Subtract 0.5 from the top
                ax.set_ylim(b, t)
                plt.subplots_adjust(left=0.25, bottom=0.25)
            if save:

                now = self.str_date_time()
                figure_name = "./graphics/" + now + '_' + str(
                    method) + '_Correlation_memory_set_' + str(
                    memory_set) + ".pdf"
                fig.savefig(figure_name, bbox_inches='tight')
                print('Correlation heatmap saved at ' + figure_name)

        self.memory[memory_set]['correlation'] = df_corr
        if threshold:
            high_corr = self.high_correlation(df_corr, threshold)
            return high_corr

    # show highest correlation values
    @staticmethod
    def high_correlation(df_corr, threshold, absolute=True):
        """ produces a list of features which exceed the threshold
        :parameter
        df_correlation: pd.DataFrame, data input
        threshold: float, threshold for correlation ouput
        **abs: bool, default=False, absolute values are evaluated

        :returns
        high_corr_pos_thresh: list, positive correlations higher the threshold
        high_corr_neg_thresh: list, negative correlations higher the threshold

        if absolute=True
        high_corr_abs_thresh: list, abs correlations higher the threshold
        """
        df_correlation = df_corr.copy()
        if not isinstance(absolute, bool):
            raise ValueError(str(absolute) + ' is not a valid boolean value ')

        if absolute:
            high_corr_abs = df_correlation.abs().unstack().sort_values(
                ascending=False).drop_duplicates()
            high_corr_abs_thresh = high_corr_abs[
                (high_corr_abs < 1) & (high_corr_abs > threshold)]

            output = high_corr_abs_thresh.round(3)
        else:
            # positive values
            high_corr_pos = df_correlation.unstack().sort_values(
                ascending=False).drop_duplicates()
            high_corr_pos_thresh = high_corr_pos[
                (high_corr_pos < 1) & (high_corr_pos > threshold)]
            # negative values
            high_corr_neg = df_correlation.unstack().sort_values(
                ascending=True).drop_duplicates()
            high_corr_neg_thresh = high_corr_neg[
                (high_corr_neg > -1) & (high_corr_pos < -threshold)]
            output = high_corr_pos_thresh.round(3), high_corr_neg_thresh.round(3)

        return output

    # highlight DataFrames True, False
    @staticmethod
    def highlight_features(data, color1='green', color2='red', **kwargs):
        """
        highlight the minimum in a Series or DataFrame
        """
        highlight = kwargs.get('highlight', False)
        color1 = 'background-color: {}'.format(color1)
        color2 = 'background-color: {}'.format(color2)
        return [color1 if v else color2 for v in highlight[data.name]]

    # show all features
    def show_features(self):
        """creates a table of all features in the memory_sets
        highlight excluded features"""
        df_data = self.df_data.copy()
        memory = self.memory
        highlight = [list(memory[i]['features'].values()) for i in memory]
        highlight = list(zip(*highlight))
        highlight = pd.DataFrame(highlight,
                                 index=df_data.columns,
                                 columns=list(memory))
        frame = [np.arange(df_data.shape[1])] * len(memory)
        frame = list(zip(*frame))
        df_features = pd.DataFrame(frame,
                                   index=df_data.columns,
                                   columns=list(memory))
        df_features.index.name = 'features'
        df_features.columns.name = 'memory set'
        self.jupyter_wrapper(df_features.style.apply(self.highlight_features, highlight=highlight))  # vertical

    # manual Features memory_set
    def select_features(self, memory_set=0, add=[], reject=[]):
        """select specific features to add or reject them from a specific
        memory_set.
        :parameter:
        df_data: DataFrame, input data to select features
        memory_set : int, past memory_set to use as basis, default=0
        reject: list of int, features to reject
        add: list of int, features to add
        """
        df_data = self.df_data.copy()
        memory = self.memory
        # feature memory_set
        last_key = max(memory.keys())

        print('memory_set={} is base!'.format(memory_set))

        # parameter checks
        if not isinstance(memory_set, int):
            raise ValueError(str(memory_set) + ' is no valid int!')
        if any(i not in range(len(df_data.columns)) for i in add):
            raise ValueError('Only positiv int < ' + str(
                len(df_data.columns)) + 'possible!')
        if any(i not in range(len(df_data.columns)) for i in reject):
            raise ValueError('Only positiv int < ' + str(
                len(df_data.columns)) + ' possible!')

        # add new_feature_memory_set based on specific memory_set
        new_feature_memory_set = deepcopy(memory[memory_set])

        # added or rejected features
        dict_add = dict.fromkeys(df_data.columns[[add]], True)
        dict_reject = dict.fromkeys(df_data.columns[[reject]], False)

        # update memory and add new memory_set
        new_feature_memory_set['features'].update(dict_reject)
        new_feature_memory_set['features'].update(dict_add)
        new_key = last_key + 1
        print('manipulated and saved as memory_set={}'.format(new_key))
        self.memory.update({new_key: new_feature_memory_set})

    def select_samples(self, memory_set=0, add=[], reject=[]):
        """select specific samples by index to add or reject them from a
        specific memory_set
        :parameter
        reject: list of str, samples to reject
        add: list of str, features to add
        memory_set : int, past memory_set to use as basis, default=0
        """
        memory = self.memory
        # feature memory_set
        last_key = max(memory.keys())

        print('memory_set={} is base!'.format(memory_set))

        # parameter checks
        if not isinstance(memory_set, int):
            raise ValueError(str(memory_set) + ' is no valid int!')

        # add new_sample_memory_set based on specific memory_set
        new_sample_memory_set = deepcopy(memory[memory_set])

        # added or rejected features
        dict_add = dict.fromkeys(add, True)
        dict_reject = dict.fromkeys(reject, False)

        # update memory and add new memory_set
        new_sample_memory_set['samples'].update(dict_reject)
        new_sample_memory_set['samples'].update(dict_add)
        new_key = last_key + 1
        print('manipulated and saved as memory_set={}'.format(new_key))
        self.memory.update({new_key: new_sample_memory_set})

    # get data set
    def get_dataset(self, memory_set):
        """ return specific data set as tuple (df_data, df_transformed)
        :parameter
        memory_set: int
        """
        df_data = self.df_data.copy()
        memory = self.memory

        if memory_set not in range(max(memory.keys())+1):
            raise ValueError('The key "' + str(memory_set) + '" does not yet exist in the memory.')

        samples = {key for key, value in memory[memory_set]['samples'].items() if value}
        features = {key for key, value in memory[memory_set]['features'].items() if value}
        df_data = df_data.loc[samples, features].copy()

        scaler = memory[memory_set]['scaler']

        if scaler is None:
            print('No transformed data')
            print('Only one file is returned')
            df_transform = None
        else:
            print('')
            print('Data was transformed with {}!'.format(scaler))
            print('Two files are returned')
            df_transform = self.normalization(df_data, scaler)
        return df_data, df_transform

    # highlight DataFrames min
    @staticmethod
    def highlight_min(data, color='blue'):
        """
        highlight the minimum in a Series or DataFrame
        """
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_min = data == data.min()
            return [attr if v else '' for v in is_min]
        else:  # from .apply(axis=None)
            is_min = data == data.minx().min()
            return pd.DataFrame(np.where(is_min, attr, ''),
                                index=data.index, columns=data.columns)

    # highlight DataFrames max
    @staticmethod
    def highlight_max(data, color='red'):
        """
        highlight the maximum in a Series or DataFrame
        """
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)

    @staticmethod
    def color(i):
        """special colors"""
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                  '#ffff33', '#a65628', '#f781bf', '#999999']
        color = to_rgb(colors[i])
        return color

    @staticmethod
    def plot_marker(scores, marker, max_clusters, ax=None, color='r'):
        """marks specific points in a plot
        :parameter
        scores: pd.DataFrame, scores
        marker:
        max_clusters
        """
        ax = ax or plt.gca()

        ax.set_xticks(np.arange(2, max_clusters + 1, 1))

        return ax.scatter(marker, scores[marker], marker='o', color=color,
                          s=5e2, alpha=0.5)

    @staticmethod
    def plot_score(scores, title, color, ax=None):
        """plots the scores"""
        ax = ax or plt.gca()

        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Score')
        ax.set_title(title)
        return ax.plot(scores, 'o-', color=color)  # ,alpha = 0.5)

    # get elbows
    @staticmethod
    def get_elbow(scores, n):
        """compute elbow """
        elbows = scores.diff()[scores.diff().diff() > 0].index.values - 1
        greatest_elbow = (scores.diff().diff()[elbows]).sort_values(
            ascending=False).head(n).index.values
        return greatest_elbow

    # plot k_scores
    def plot_k_scores(self, memory_set, save=False):
        """plots all scores of one or multiple specific memory_sets
        :parameter
        memory_set: int or list of int
        save: bool, saves figure
        """
        memory = self.memory

        if isinstance(memory_set, int):
            memory_set = [memory_set]
            # number of elbows/min/max
            n = 1
        else:
            n = 1

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # text box positions
        # pos_x = 2/3 * max_clusters
        # pos_y = scores['scores'].max()

        # n   # number of elbows/min/max
        n = 1
        handles = []
        for j, i in enumerate(memory_set):
            scores = memory[i]['scores']
            max_clusters = max(scores.index)

            # upper left plot
            handles += self.plot_score(scores['ssr-score'],
                                       'Sum of Squared Residuals',
                                       color=self.color(j),
                                       ax=ax[0][0])
            # calc elbows
            elbow = self.get_elbow(scores['ssr-score'], n)
            self.plot_marker(scores['ssr-score'],
                             elbow,
                             max_clusters,
                             ax=ax[0][0],
                             color=self.color(j))

            # upper right plot
            self.plot_score(scores['silhouette-score'],
                            'Silhouette Score',
                            color=self.color(j),
                            ax=ax[0][1])
            maxima_sil = scores['silhouette-score'][1:-1].sort_values(
                ascending=False).head(n).index.values
            self.plot_marker(scores['silhouette-score'],
                             maxima_sil,
                             max_clusters,
                             ax=ax[0][1],
                             color=self.color(j))

            # lower left plot
            self.plot_score(scores['calinski-score'],
                            'Calinski-Harabaz Score',
                            color=self.color(j),
                            ax=ax[1][0])
            maxima_cal = scores['calinski-score'][1:-1].sort_values(
                ascending=False).head(n).index.values
            self.plot_marker(scores['calinski-score'],
                             maxima_cal,
                             max_clusters,
                             ax=ax[1][0],
                             color=self.color(j))

            # lower right plot
            self.plot_score(scores['bouldin-score'],
                            'Davies-Bouldin Score',
                            self.color(j),
                            ax=ax[1][1])
            minima_bol = scores['bouldin-score'][1:-1].sort_values(
                ascending=True).head(n).index.values
            self.plot_marker(scores['bouldin-score'],
                             minima_bol,
                             max_clusters,
                             ax=ax[1][1],
                             color=self.color(j))

        # plt.rcParams['axes.grid'] = True
        names = ['memory_set: {}'.format(i) for i in memory_set]
        for ax in fig.axes:
            ax.legend(handles, names, frameon=False)
            ax.grid()
        # subplots_adjust
        fig.subplots_adjust(hspace=0.3)

        if save:
            now = self.str_date_time()
            figname = './graphics/{}_Scoreplots_memory_set_{}.pdf'.format(
                now, str(memory_set))
            fig.savefig(figname, bbox_inches='tight')
            print('Scoreplots saved as ' + figname)

        # Infos for Plots
        print('-------------------------------')
        print('Indicators for the most likely optimal number of "k"')
        print('-------------------------------')
        print('Least-squares : Find the Elbow ')
        print('Silhouett-Score : the maximum value is the optimal one')
        print('Calinski-Harabaz-Score : the maximum value is the optimal one')
        print('Davies-Bouldin-Score : the minimum value is the optimal one')
        print('-------------------------------')

        plt.show()

    # K-Means mit variabler Clusteranzahl
    def optimal_k(self, memory_set, kmax, kmin=2, scaler='standard',
                  plot=True, new=False):
        """ compute multiple scores to find out about the optimal k
        k is variated from kmin to kmax
        the scaler is saved to the memory. If you want to change the scaler,
        you should use 'new=True' to generate a new memory_set
        :parameter
        memory_set: int,
        kmax: int,
        kmin: int,
        scaler: str, 'standard', 'minmax', 'robust', 'None'
        plot: bool
        new: bool, creates a new memory_set
        """
        # cluster_algorithmus von blonde als parameter uebergeben
        # mit voreingestelltem k
        # k ueber clustering.k=k nachstellen

        if kmin < 2:
            raise ValueError('kmin needs to be greater than 1')

        df_data = self.df_data.copy()
        memory = self.memory

        last_key = max(memory.keys())
        if memory_set not in range(last_key + 1):
            raise ValueError('The key does not yet exist in the memory.')

        # get dataset
        samples = {key for key, value in self.memory[memory_set]['samples'].items() if
                   value}
        features = {key for key, value in self.memory[memory_set]['features'].items() if
                    value}
        df_data = df_data.loc[samples, features].copy()
        if scaler:
            df_transform = self.normalization(df_data, scaler)
        else:
            df_transform = df_data.copy()
        if new:
            new_sample_memory_set = deepcopy(self.memory[memory_set])
            memory_set = last_key + 1
            self.memory.update({memory_set: new_sample_memory_set})
            print('New memory set ' + str(memory_set) + ' was added!')

        self.memory[memory_set]['scaler'] = scaler
        print('data set was normalized with ' + str(scaler) + '-scaler.')
        # initiate scores
        score_ssr = []
        score_silhouette = []
        score_calinski = []
        score_bouldin = []

        for k in range(kmin, kmax + 1):
            clustering = KMeans(n_clusters=k, n_init=50)
            clustering.fit(df_transform)

            # compute scores
            score_ssr.append(clustering.inertia_)
            score_silhouette.append(
                silhouette_score(df_transform, clustering.labels_,
                                 metric='euclidean'))
            score_calinski.append(
                calinski_harabasz_score(df_transform, clustering.labels_))
            score_bouldin.append(
                davies_bouldin_score(df_transform, clustering.labels_))
        # crate dataframe
        scores = np.array([score_ssr, score_silhouette, score_calinski,
                     score_bouldin])
        columns = ['ssr-score', 'silhouette-score', 'calinski-score',
                   'bouldin-score']
        index = list(range(2, kmax + 1))
        df_scores = pd.DataFrame(data=scores.T, columns=columns, index=index)
        df_scores.index.set_names('cluster', inplace=True)
        df_scores['looped'] = None

        self.memory[memory_set]['scores'] = df_scores

        if plot:
            self.plot_k_scores(memory_set)

    # highlight DataFrames True
    @staticmethod
    def highlight_True(data, color='green'):
        """highlight the True in a Series or DataFrame"""
        attr = 'background-color: {}'.format(color)
        return [attr if v else '' for v in data]

    # highlight DataFrames False
    @staticmethod
    def highlight_False(data, color='red'):
        """highlight the False in a Series or DataFrame"""
        attr = 'background-color: {}'.format(color)
        return [attr if not v else '' for v in data]

    # label mapping
    @staticmethod
    def label_mapping(df_labels, k_cluster, loop):
        """maps the labels of two kmeans interations
        sets of the samples in the clusters are compared
         cluster with the highest amount of symmetrie are mapped.
         doesnt work to good if cluster are very small

         if cluster split up in 2 clusters of same size, the mapping is
         incorrect"
         """
        mapping_correction = {}

        # compare labels and creat mapping to have similar labels
        for label in range(k_cluster):
            mapping_comparisson = df_labels.groupby(loop)[0].value_counts()
            mapping_comparisson_matrix = mapping_comparisson.unstack()

            first_label = mapping_comparisson_matrix[label].idxmax()
            # check if this label assignement does have the most members
            if mapping_comparisson_matrix.loc[first_label, :].idxmax() == \
                    label:
                mapping_correction.update({first_label: label})

        # reassign splitted clusters
        count = 0
        while len(mapping_correction) < k_cluster:

            free_labels = set(range(k_cluster)).difference(set(
                mapping_correction.keys()))
            lost_label = set(range(k_cluster)).difference(set(
                mapping_correction.values())).pop()
            #NEW
            mapping_comparisson = df_labels.groupby(loop)[0].value_counts()
            mapping_comparisson_matrix = mapping_comparisson.unstack()
            free_label = mapping_comparisson_matrix.loc[free_labels,
                                                        lost_label].idxmax()
            if free_label > 0:
                mapping_correction.update({free_label: lost_label})
            else:
                any_label = set(range(k_cluster)).difference(set(
                    mapping_correction.keys())).pop()
                mapping_correction.update({any_label: lost_label})
                print('Loop {}'.format(str(loop)))
                print('Did not find a trivial solution for the mapping.')
                print('Label {} was reassigned afterwards and '
                      'might not be correct!'.format(any_label))
            if count > 50:
                print('Loop {}'.format(str(loop)))
                print('Label mapping of this loop will be wrong! Had to '
                      'break loop. This loop has to be excluded!')
                break
            count += 1
        df_labels[loop].replace(mapping_correction, inplace=True)

        return df_labels

    # loop_clustering
    def loop_clustering(self, memory_set, k_cluster, max_loops, scaler=None,
                        n_init=10):
        """ loops kmeans for a specific numbers of clusters.
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of clusters to be identified
        max_loops: int, number of loops for kmeans
        scaler: str, transformer to be used, default= taken from memory
        n_init: int, Number iterations k-means algorithm will be run with
        different centroid seeds,

        for mor info check scikit-learn:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        """
        df_data = self.df_data.copy()
        memory = self.memory
        # info = kwargs.get('info', False)

        # parameter check
        if not isinstance(memory_set, (int, bool)):
            raise ValueError(str(memory_set) + ' is no valid int/bool!')

        samples = {key for key, value in memory[memory_set]['samples'].items()
                   if value}
        features = {key for key, value in memory[memory_set]['features'].items()
                    if value}
        df_data = df_data.loc[samples, features].copy()

        if scaler:
            print('Scaler was taken from parameter.')
            print('Scaler is' + str(scaler))
        else:
            scaler = memory[memory_set]['scaler']
            print('Scaler was taken from memory.')
        if scaler:
            df_transform = self.normalization(df_data, scaler)
        else:
            df_transform = df_data.copy()

        print('data set was normalized with {}'.format(scaler))
        df_labels = pd.DataFrame(index=df_data.index)
        score_ssr = []
        score_silhouette = []
        score_calinski = []
        score_bouldin = []

        for loop in range(max_loops):

            # K-means Berechnung nach Auswahl der optimalen Cluster
            kmeans_l = KMeans(n_clusters=k_cluster, n_init=n_init).fit(
                df_transform)

            # Labels
            df_labels[loop] = kmeans_l.labels_

            # mapping if more then one loop
            if loop > 0:
                # correct cluster label mapping
                df_labels = self.label_mapping(df_labels, k_cluster, loop)

            # Scores
            score_ssr.append(
                kmeans_l.inertia_)
            score_silhouette.append(
                silhouette_score(df_data, kmeans_l.labels_,
                                 metric='euclidean'))
            score_calinski.append(
                calinski_harabasz_score(df_data, kmeans_l.labels_))
            score_bouldin.append(
                davies_bouldin_score(df_data, kmeans_l.labels_))

        scores = np.array([score_ssr, score_silhouette, score_calinski,
                     score_bouldin])
        columns = ['ssr-score', 'silhouette-score', 'calinski-score',
                   'bouldin-score']
        index = list(range(max_loops))
        scores = pd.DataFrame(data=scores.T, columns=columns, index=index)
        scores.index.set_names('loop', inplace=True)
        scores.columns.name = 'Scores'

        # min/max highlighting
        self.jupyter_wrapper(scores.round(5).style.apply(
            self.highlight_max, subset=['silhouette-score', 'calinski-score']).apply(
            self.highlight_min, subset=['ssr-score', 'bouldin-score']))
        # manipulate scores
        self.memory[memory_set]['scores']['looped'].at[k_cluster] = max_loops

        # manipulate labels
        self.memory[memory_set]['labels'].update({k_cluster: df_labels})

    def show_scores(self, memory_set):
        """show specific scores
        :parameter
        memory_set: int, memory_set to be used"""
        self.jupyter_wrapper(self.memory[memory_set]['scores'])

    def cluster_sizes(self, memory_set, k_cluster):
        """compute the cluster sizes of the clustering of a specific memory
        set with a specific number of clusters
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of clusters of the clustering
        """

        df_labels = self.memory[memory_set]['labels'][k_cluster].copy()
        # drop all duplicated results
        df_labels = df_labels.T.drop_duplicates().T

        df_cluster_sizes = pd.DataFrame()
        for column in df_labels.columns:
            df_cluster_sizes.loc[:, column] = df_labels.loc[:, column].value_counts()
        df_cluster_sizes.index.name = 'cluster'
        df_cluster_sizes.columns.name = 'loop'
        df_cluster_sizes.sort_index(inplace=True)
        cmap = cm.get_cmap(self.cmap_col)
        self.jupyter_wrapper(df_cluster_sizes.style.background_gradient(
            cmap=cmap, axis=1))
        return df_cluster_sizes

    def borderliner(self, memory_set, k_cluster, exclude=None):
        """identify the samples which jump clusters
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        exclude: int or list of int, these loops will be excluded"""
        df_transfer = self.df_transfer.copy()
        memory = self.memory

        df_labels = memory[memory_set]['labels'][k_cluster]
        # drop certain loops
        if exclude:
            df_labels.drop(columns=exclude, inplace=True)
        # Borderliner
        different_labels = (df_labels.diff(
            axis=1, periods=1).iloc[:, 2:] != 0).sum(
            axis=1)#.sort_values(ascending=False)
        jumper = different_labels[different_labels != 0].index
        # Cluster jumps
        jumps = pd.Series(data=list(map(set, df_labels.loc[jumper].values)),
                          index=jumper).rename('Cluster jump')

        # mapping
        new_order_feat = df_labels.loc[jumper].sum().sort_values().index
        new_order_samp = df_labels.loc[jumper].sum(axis=1).sort_values().index
        jumper_name = df_transfer['Name'].loc[new_order_samp]#.rename('Name')

        # Cluster coloring
        df_borderliner_sorted = df_labels.loc[new_order_samp, new_order_feat].copy()
        df_borderliner_info = pd.DataFrame([jumper_name, jumps]).transpose()

        self.memory[memory_set]['borderliner'][k_cluster] = df_borderliner_sorted.copy()
        print("The following table shows all sample which are not allocated"
              " to the same cluster every time.")
        self.jupyter_wrapper(df_borderliner_info)
        return df_borderliner_sorted

    def plot_silhouettes(self, memory_set, k_cluster, loop, save=False,
                         output=False, transform=True):
        """plots the silhouette scores for every samples of a computed
        clustering

        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of clusters used in the clustering
        loop: int, specific loop which is selected"""


        df_data, df_transform = self.get_dataset(memory_set)
        if transform:
            data = df_transform.copy()
        else:
            data = df_data
        cluster_labels = self.get_labels(memory_set, k_cluster, loop)

        # Compute the average silhouette score
        silhouette_avg = silhouette_score(data.values, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data.values,
                                                      cluster_labels)
        # Create Plot
        fig, ax1 = plt.subplots()
        fig.set_size_inches(7, 7)

        y_lower = 10
        for i in range(k_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            # get colormap
            cmap = cm.get_cmap(self.cmap_col)
            color = cmap(float(i) / k_cluster)
            # plot area plots
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for K-Means "
                      "with k_cluster = %d" % k_cluster),
                     fontsize=14, fontweight='bold')

        plt.show()
        print("For k_cluster =", k_cluster,
              "The average silhouette_score is :", silhouette_avg)
        if save:
            # plt.axis('equal')
            now = self.str_date_time()
            figname = './graphics/{}_Silhouette_' \
                      'analysis_n-{}.pdf'.format(now, k_cluster)
            fig.savefig(figname, bbox_inches='tight')
            print('Silhouette saved as ' + figname)
        if output:
            df_output = pd.DataFrame(data=sample_silhouette_values,
                                     index=data.index,
                                     columns=['Silhouette'])
            df_output = df_output.join(cluster_labels.rename('Cluster'))

            return df_output

    def get_labels(self, memory_set, k_cluster, loop):
        """returns Series of specific labels
        :parameter
        mmemory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected"""
        labels = self.memory[memory_set]['labels'][k_cluster][loop].copy()
        return labels

    def get_clustering(self, memory_set, k_cluster, loop):
        """returns data set with specific cluster labels
         :parameter
        mmemory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected
        """
        memory = self.memory.copy()
        df_data = self.df_data.copy()
        samples = {key for key, value in memory[memory_set][
            'samples'].items()
                   if value}
        features = {key for key, value in
                    memory[memory_set]['features'].items()
                    if value}
        df_data = df_data.loc[samples, features].copy()
        labels = self.memory[memory_set]['labels'][k_cluster][loop].copy()
        names = self.df_transfer.Name.copy()

        df_clustering = df_data.join([labels.rename('Cluster'), names])
        return df_clustering

    @staticmethod
    def cmap_discretize(cmap, N):
        """Return a discrete colormap from the continuous colormap cmap.
            cmap: colormap instance, eg. cm.jet.
            N: number of colors.
        """
        #if type(cmap) == str:
        #    cmap = get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
        colors_rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N + 1)
        cdict = {}
        for ki, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [
                (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i
                in range(N + 1)]
        # Return colormap object.
        return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

    @staticmethod
    def my_colors(k):
        """special colors"""
        colors = [
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
            (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
            (1.0, 0.4980392156862745, 0.0),
            (1.0, 1.0, 0.2),
            (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
            (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
            (0.6, 0.6, 0.6)]
        return colors[:k]

    def plot_cluster(self, memory_set, k_cluster, loop, save=False):
        """plots a map of the computed clustering.
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected
        save: bool, saves the figure"""

        # name series
        labels = self.memory[memory_set]['labels'][k_cluster][loop].copy()
        labels.name = 'cluster'

        # organize gdf
        path_map = os.getcwd()
        for root, dirs, files in os.walk(path_map):
            for file in files:
                if file.endswith('NUTS3.shp'):
                    path2 = root + '/' + file
        gdf_map_all = gpd.read_file(path2, encoding='utf-8')
        gdf_map_all.drop(index=gdf_map_all.index[gdf_map_all.GF == 2],
                         inplace=True) # GF 2 = zusaetzliche Gewaesser entfernen
        gdf_map = gdf_map_all.merge(labels, how='inner', left_on='NUTS_CODE',
                                right_index=True)

        cmap = cm.get_cmap(self.cmap_col)
        cmd = self.cmap_discretize(cmap, N=k_cluster)

        # plot gdf
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_map_all.plot(ax=ax, color='silver', edgecolor='black')
        gdf_map.plot(ax=ax, column='cluster', cmap=cmd, edgecolor='black')  # , legend=True)

        # manually create colorbar
        vmin, vmax = 0, k_cluster
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap=cmd,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # fake up the array of the scalar mappable.
        sm._A = []

        ticks = ['Cluster {}'.format(i) for i in range(k_cluster)]
        cbar = fig.colorbar(sm, cax=cax, ticks=range(k_cluster))
        cbar.ax.set_yticklabels(ticks)
        ax.set_axis_off()
        fig.suptitle('Identified Cluster in Germany')
        plt.show()

        if save:
            # plt.axis('equal')
            now = self.str_date_time()
            figname = './graphics/{}_Clustermap_n-{}.png'.format(now,
                                                               str(k_cluster))
            fig.savefig(figname, bbox_inches='tight')
            print('Clustermap saved as ' + figname)

    def plot_nuts3(self, nuts3, save=False):
        """plots a map of specific NUTS3"
        :parameter
        nuts3: list of str, nuts3 to be colored in the map
        save: bool, saves the figure"""

        # organize gdf
        path_map = os.getcwd()
        for root, dirs, files in os.walk(path_map):
            for file in files:
                if file.endswith('NUTS3.shp'):
                    path2 = root + '/' + file
        gdf_map = gpd.read_file(path2, encoding='utf-8')
        gdf_map.drop(index=gdf_map.index[gdf_map.GF == 2],
                     inplace=True)  # GF 2 = zusaetzliche Gewaesser entfernen
        gdf_map.set_index('NUTS_CODE', inplace=True)
        gdf_map_selected = gdf_map.loc[nuts3]

        n_districts = gdf_map_selected.shape[0]
        cbar = True
        if n_districts <= 20:
            cmap = cm.get_cmap(self.cmap_col)

        else:
            cbar = False
            cmap = cm.get_cmap("Spectral")
            color = cmap(float(1) / 12)

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_map.plot(ax=ax, color='silver', edgecolor='black')
        if cbar:
            gdf_map_selected.plot(ax=ax, column='NUTS_NAME',
                                  cmap=cmap, edgecolor='black')  # , legend=True)

            # manually crate colorbar
            vmin, vmax = 0, n_districts
            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,
                                                                   vmax=vmax))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []

            tick_names = [i for i in gdf_map_selected.NUTS_NAME]
            cbar = fig.colorbar(sm, cax=cax, ticks=range(n_districts))
            cbar.ax.set_yticklabels(tick_names)
        else:
            gdf_map_selected.plot(ax=ax, color=color)
        ax.set_axis_off()
        fig.suptitle('Selected districts in Germany')
        plt.show()

        if save:
            # plt.axis('equal')
            now = self.str_date_time()
            figname = './graphics/{}_District_map_n-{}.png'.format(
                now, str(n_districts))
            fig.savefig(figname, bbox_inches='tight')
            print('District map saved as ' + figname)

    def ident_typregions(self, memory_set, k_cluster, loop):
        """ identify the typregions of a specific clustering
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected
        """
        memory = self.memory.copy()
        df_data = self.df_data.copy()
        df_transfer = self.df_transfer.copy()

        labels = self.memory[memory_set]['labels'][k_cluster][loop].copy()

        samples = {key for key, value in memory[memory_set][
            'samples'].items()
                   if value}
        features = {key for key, value in
                    memory[memory_set]['features'].items()
                    if value}
        df_data = df_data.loc[samples, features].copy()

        labels.name = 'cluster'
        df_data = df_data.join(labels, how='inner')
        df_center = df_data.groupby('cluster').median()

        # ordinary least squares
        for index, feature in df_data.iterrows():
            sse = np.sum((feature.drop('cluster').values - df_center.loc[
                feature.cluster.astype(int)]) ** 2)
            df_data.at[index, 'SSE'] = sse

        typ_index = df_data.groupby('cluster').SSE.idxmin()
        typ_index.name = 'NUTS3'
        df_typregions = pd.merge(typ_index,
                                  df_transfer['Name'].rename('Typregion'),
                                  how='inner', left_on='NUTS3',
                                  right_index=True)
        #self.jupyter_wrapper(df_typregions)
        return (df_typregions, df_data)


    def create_memory_set(self, nuts3):
        """ manually create memory_set  by nuts3 index
        :parameter
        nuts3: list, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected
        """

        memory = self.memory
        # create new memory_set
        dict_new = dict.fromkeys(self.df_transfer.index, False)
        # add nuts3 as ture
        dict_true = dict.fromkeys(nuts3, True)
        dict_new.update(dict_true)

        # update memory_set
        new_key = max(memory.keys()) + 1
        # new dict_entry = copy of initialisation
        new_sample_memory_set = deepcopy(memory[memory_set])
        new_sample_memory_set['samples'].update(dict_new)
        new_sample_memory_set['outlier'] = (memory_set, pos_threshold,
                                            neg_threshold)
        self.memory.update({new_key: new_sample_memory_set})

        print('created new memory_set={}'.format(new_key))


    def plot_typregions(self, memory_set, k_cluster, loop, save=False):
        """ plots the typregions of a specific clustering
        :parameter
        memory_set: int, memory_set to be used
        k_cluster: int, number of cluster used in the clustering
        loop: int, specific loop which is selected
        save: bool, saves the figure"""

        typregions = self.ident_typregions(memory_set, k_cluster, loop)
        typregions = typregions.reset_index()
        typregions.set_index('NUTS3', inplace=True)


        # organize gdf
        path_map = os.getcwd()
        for root, dirs, files in os.walk(path_map):
            for file in files:
                if file.endswith('NUTS3.shp'):
                    path2 = root + '/' + file
        gdf_map = gpd.read_file(path2, encoding='utf-8')
        gdf_map.drop(index=gdf_map.index[gdf_map.GF == 2],
                     inplace=True)  # GF 2 = zusaetzliche Gewaesser entfernen
        gdf_map_selection = gdf_map.merge(typregions, how='inner',
                                          left_on='NUTS_CODE',
                                          right_index=True)

        # create colormap
        n_cluster = gdf_map_selection.cluster.nunique()
        cmap = cm.get_cmap(self.cmap_col)
        cmd = self.cmap_discretize(cmap, N=n_cluster)
        # plot gdf
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_map.plot(ax=ax, color='silver', edgecolor='black')
        gdf_map_selection.plot(ax=ax, column='cluster', cmap=cmd, edgecolor='black')  # , legend=True)

        # manually create colorbar
        vmin, vmax = 0, n_cluster
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap=cmd,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # fake up the array of the scalar mappable.
        sm._A = []

        # ticks_name = [i for i in gdf_map_selection.NUTS_NAME]
        ticks_name = typregions.Typregion.to_list()
        cbar = fig.colorbar(sm, cax=cax, ticks=range(n_cluster))
        cbar.ax.set_yticklabels(ticks_name)
        ax.set_axis_off()
        fig.suptitle('Identified Typregions')
        plt.show()

        if save:
            # plt.axis('equal')
            now = self.str_date_time()
            figname = './graphics/{}_Typregionsmap_n-{}.png'.format(now,
                                                            str(n_cluster))
            fig.savefig(figname, bbox_inches='tight')
            print('Clustermap saved as ' + figname)
