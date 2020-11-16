import pandas as pd
import numpy as np
import os
import matplotlib

from matplotlib import pyplot as plt


class ExportPlotsToExcel():
    """
    Class to export plots to excel row by row

    xlsx = ExportPlotsToExcel()
    """
    def __init__(self):
        """
        Initialise excel workbook and setup default format
        see generate plots.xlsx
        """
        self.path = 'plots.xlsx'
        self.writer = pd.ExcelWriter(self.path, engine='xlsxwriter')
        self.workbook  = self.writer.book
        self.worksheet = self.workbook.add_worksheet("PLOTS")

       
        column_format = self.workbook.add_format({'text_wrap': True})
        self.worksheet.set_column('A:A', 60,column_format)

        self.worksheet.set_column('B:B', 135)

        column_format = self.workbook.add_format({'text_wrap': True})
        column_format.set_align('top')
        self.worksheet.set_column('F:F', 60,column_format)

        bold_cell_format = self.workbook.add_format({'bold': True,'color':'white'})
        bold_cell_format.set_align('center')
        bold_cell_format.set_align('vcenter')
        bold_cell_format.set_bg_color('green')
        self.worksheet.write('A1', 'Feature',bold_cell_format)
        self.worksheet.write('B1', 'Plots',bold_cell_format)
        self.worksheet.write('C1', 'Max Prob',bold_cell_format)
        self.worksheet.write('D1', 'Min Prob',bold_cell_format)
        self.worksheet.write('E1', 'Max-Min',bold_cell_format)
        self.worksheet.write('F1', 'Notes',bold_cell_format)

        self.plain_cell_format = self.workbook.add_format()
        self.plain_cell_format.set_align('center')
        self.plain_cell_format.set_align('vcenter')

    def save_plot(self,index,train_test,image,mx,mn,mx_mn,has_test=True):
        """
        Saves plots images into excel sheet named "PLOTS" row by row

        :param index: row number
        :param train_test: 'Train' or 'Test' 
        :param image: plot image
        :param mx: maximum of target_col mean
        :param mn: minimum of target_col mean
        :param mx_mn: max-min indicates feature separability power
        :param has_test: controls row_size
        :return: new row with plot
        """
        if has_test:
            row_size=600
        else:
            row_size=300
        self.worksheet.set_row(index-1, row_size)
        self.worksheet.write('A{}'.format(index), train_test,self.plain_cell_format)
        self.worksheet.insert_image('B{}'.format(index), image,{'x_scale': 0.8, 'y_scale': 0.8,'x_offset': 14, 'y_offset': 10})
        self.worksheet.write('C{}'.format(index), mx,self.plain_cell_format)
        self.worksheet.write('D{}'.format(index), mn,self.plain_cell_format)
        self.worksheet.write('E{}'.format(index), mx_mn,self.plain_cell_format)


def get_grouped_data(input_data, feature, target_col, bins, cuts=0):
    """
        Bins continuous features into equal sample size buckets and 
        returns the target mean in each bucket. Separates out nulls into 
        another bucket.
        
        :param input_data: dataframe containg features and target column.
        :param feature: feature column name.
        :param target_col: target column.
        :param bins: Number bins required.
        :param cuts: if buckets of certain specific cuts are required. Used 
        on test data to use cuts from train.
        :return: If cuts are passed only grouped data is returned, else cuts 
        and grouped data is returned.
    """

    input_data[feature] = input_data[feature].round(5)
    has_null = pd.isnull(input_data[feature]).sum() > 0
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts == 0:
        is_train = 1
        prev_cut = min(input_data[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(input_data[feature], i * 100 / bins)
            if (
                next_cut > prev_cut + 0.000001
            ):  # float numbers shold be compared with some threshold!
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        # if reduced_cuts>0:
            # print(
            # 'Reduced the number of bins due to less variation in feature'
            # )
        
        cut_series = pd.cut(input_data[feature], cuts)
    else:
        cut_series = pd.cut(input_data[feature], cuts)

    grouped = input_data.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]}
    )
    grouped.columns = [
        "_".join(cols).strip() for cols in grouped.columns.values
    ]
    grouped[grouped.index.name] = grouped.index
    grouped.reset_index(inplace=True, drop=True)
    grouped = grouped[[feature] + list(grouped.columns[0:3])]
    grouped = grouped.rename(
        index=str, columns={target_col + "_size": "Samples_in_bin"}
    )
    grouped = grouped.reset_index(drop=True)
    corrected_bin_name = (
        "["
        + str(round(min(input_data[feature]), 5))
        + ", "
        + str(grouped.loc[0, feature]).split(",")[1]
    )
    grouped[feature] = grouped[feature].astype("category")
    grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
    grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype("category")
        grouped_null[feature] = grouped_null[feature].cat.add_categories(
            "Nulls"
        )
        grouped_null.loc[0, feature] = "Nulls"
        grouped_null.loc[0, "Samples_in_bin"] = len(data_null)
        grouped_null.loc[0, target_col + "_mean"] = data_null[
            target_col
        ].mean()
        grouped_null.loc[0, feature + "_mean"] = np.nan
        grouped[feature] = grouped[feature].astype("str")
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    grouped[feature] = grouped[feature].astype("str").astype("category")
    if is_train == 1:
        return (cuts, grouped)
    else:
        return grouped


def draw_plots(input_data, feature, target_col,trend_correlation=None,show_plots=True,export_to_excel=False):
    """
        Draws univariate dependence plots for a feature.
        :param input_data: grouped data contained bins of feature and 
        target mean.
        :param feature: feature column name.
        :param target_col: target column.
        :param trend_correlation: correlation between train and test trends 
        of feature wrt target.
        :param show_plots: if to show plots on the screen (stdout) - they can be so many , 1000 features
        :param export_to_excel: if to generate excel report with plots instead
        :return: Draws trend plots for feature or save figure for export
    """
    train_input_data,test_input_data = input_data
    nrows=2
    data = {1:train_input_data,2:train_input_data,3:test_input_data,4:test_input_data}

    has_test = type(test_input_data) == pd.core.frame.DataFrame

    if has_test==False:
        nrows = 1
        
    fig, big_axes = plt.subplots( figsize=(12.0, 5.0 if nrows==1 else 10.0) , nrows=nrows, ncols=1, sharey=True) 
    big_axes=[big_axes] if nrows==1 else big_axes

    for row, big_ax in enumerate(big_axes, start=1):
        if row==1:
            big_ax.set_title("{} \n" .format("Train"), fontsize=16)
        elif row==2:
            big_ax.set_title("{} \n" .format("Test"), fontsize=16)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False


    for i in range(1,(nrows*2)+1):
        ax = fig.add_subplot(nrows,2,i)
        if i==1 or i==3:
            input_data=data[i]
            trend_changes = get_trend_changes(
                grouped_data=input_data, feature=feature, target_col=target_col
            )
            ax.plot(input_data[target_col + "_mean"], marker="o")
            ax.set_title("Average of " + target_col + " wrt " + feature)
            ax.set_xticks(np.arange(len(input_data)))
            ax.set_xticklabels((input_data[feature]).astype("str"))
            plt.xticks(rotation=45)
            ax.set_xlabel("Bins of " + feature)
            ax.set_ylabel("Average of " + target_col)
            comment = "Trend changed " + str(trend_changes) + " times"
            if trend_correlation == 0 and i==3:
                comment = comment + "\n" + "Correlation with train trend: NA"
            elif trend_correlation != None and i==3:
                comment = (
                    comment
                    + "\n"
                    + "Correlation with train trend: "
                    + str(int(trend_correlation * 100))
                    + "%"
                )

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
            ax.text(
                0.05,
                0.95,
                comment,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
                transform=ax.transAxes,
            )
        elif i==2 or i==4:
            input_data=data[i]
            ax.bar(
                np.arange(len(input_data)), input_data["Samples_in_bin"], alpha=0.5
            )
            ax.set_xticks(np.arange(len(input_data)))
            ax.set_xticklabels((input_data[feature]).astype("str"))
            plt.xticks(rotation=45)
            ax.set_xlabel("Bins of " + feature)
            ax.set_ylabel("Bin-wise sample size")
            ax.set_title("Samples in bins of " + feature)


    fig.set_facecolor('w')
    plt.tight_layout()

    #optional to export plots to excel
    if export_to_excel:
        plt.savefig('{}.png'.format(feature))

    #optional to show plots on console
    if show_plots:
        plt.show()
    
    # release resources
    if show_plots==False:
        plt.close('all')

    

def get_trend_changes(grouped_data, feature, target_col, threshold=0.03):
    """
        Calculates number of times the trend of feature wrt target changed
        direction.
        :param grouped_data: grouped dataset.
        :param feature: feature column name.
        :param target_col: target column.
        :param threshold: minimum % difference required to count as trend 
        change.
        :return: number of trend chagnes for the feature.
    """

    grouped_data = grouped_data.loc[
        grouped_data[feature] != "Nulls", :
    ].reset_index(drop=True)
    target_diffs = grouped_data[target_col + "_mean"].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = (
        grouped_data[target_col + "_mean"].max()
        - grouped_data[target_col + "_mean"].min()
    )
    target_diffs_mod = target_diffs.fillna(0).abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return tot_trend_changes


def get_trend_correlation(grouped, grouped_test, feature, target_col):
    """
        Calculates correlation between train and test trend of feature 
        wrt target.
        
        :param grouped: train grouped data.
        :param grouped_test: test grouped data.
        :param feature: feature column name.
        :param target_col: target column name.
        :return: trend correlation between train and test.
    """

    grouped = grouped[grouped[feature] != "Nulls"].reset_index(drop=True)
    grouped_test = grouped_test[grouped_test[feature] != "Nulls"].reset_index(
        drop=True
    )

    if grouped_test.loc[0, feature] != grouped.loc[0, feature]:
        grouped_test[feature] = grouped_test[feature].cat.add_categories(
            grouped.loc[0, feature]
        )
        grouped_test.loc[0, feature] = grouped.loc[0, feature]
    grouped_test_train = grouped.merge(
        grouped_test[[feature, target_col + "_mean"]],
        on=feature,
        how="left",
        suffixes=("", "_test"),
    )
    nan_rows = pd.isnull(grouped_test_train[target_col + "_mean"]) | pd.isnull(
        grouped_test_train[target_col + "_mean_test"]
    )
    grouped_test_train = grouped_test_train.loc[~nan_rows, :]
    if len(grouped_test_train) > 1:
        trend_correlation = np.corrcoef(
            grouped_test_train[target_col + "_mean"],
            grouped_test_train[target_col + "_mean_test"],
        )[0, 1]
    else:
        trend_correlation = 0
        print(
            "Only one bin created for "
            + feature
            + ". Correlation can't be calculated"
        )

    return trend_correlation


def univariate_plotter(feature, data, target_col, bins=10, data_test=0,plot_number=0,show_plots=True,export_to_excel=False,xlsx=None):
    
  
    """
        Calls the draw plot function and editing around the plots.
        
        :param feature: feature column name.
        :param data: dataframe containing features and target columns.
        :param target_col: target column name.
        :param bins: number of bins to be created from continuous feature.
        :param data_test: test data which has to be compared with input data 
        for correlation.
        :param plot_number: each test/train plot is assigned a number for tracking purposes and saving into respective excel row number
        :param show_plots: optional to show plots on the screen (stdout) - they can be so many , 1000 features
        :param export_to_excel: if to generate excel report with plots instead
        :param xlsx: excel export class object
        :return: grouped data if only train passed, else (grouped train 
        data, grouped test data).
    """

    #make it optional to show plots to console
    if show_plots:
        print(" {:^100} ".format("Plots for " + feature))
    

    if data[feature].dtype == "O":
        print("Categorical feature not supported")
    else:
        cuts, grouped = get_grouped_data(
            input_data=data, feature=feature, target_col=target_col, bins=bins
        )
        has_test = type(data_test) == pd.core.frame.DataFrame

    

        if has_test:
            grouped_test = get_grouped_data(
                input_data=data_test.reset_index(drop=True),
                feature=feature,
                target_col=target_col,
                bins=bins,
                cuts=cuts,
            )
            trend_corr = get_trend_correlation(
                grouped, grouped_test, feature, target_col
            )
           
            

            """
                refactored input_data to tuple type (train_grouped, test_grouped)
                calls draw_plots function once
                for ploting train/test as one 2x4 grid
            """
            _input_data = (grouped,grouped_test)

            draw_plots(
                input_data=_input_data,
                feature=feature,
                target_col=target_col,
                trend_correlation=trend_corr,
                show_plots=show_plots,
                export_to_excel=export_to_excel
            )

            max_prob = grouped[target_col + "_mean"].max()
            min_prob = grouped[target_col + "_mean"].min()

            if export_to_excel:
                xlsx.save_plot(index=plot_number+1,train_test=feature,image='{}.png'.format(feature),mx=max_prob,mn=min_prob,mx_mn=max_prob-min_prob)

        else:
            """
                refactored input_data to tuple type (train_grouped, None) - no test data provided
                train is mandatory
                calls draw_plots function once
                for ploting train/test as one 2x2 grid
            """
            _input_data = (grouped,None)

            draw_plots(
                input_data=_input_data, feature=feature, target_col=target_col,show_plots=show_plots,export_to_excel=export_to_excel
            )

            max_prob = grouped[target_col + "_mean"].max()
            min_prob = grouped[target_col + "_mean"].min()

            if export_to_excel:
                xlsx.save_plot(index=plot_number+1,train_test=feature,image='{}.png'.format(feature),mx=max_prob,mn=min_prob,mx_mn=max_prob-min_prob,has_test=has_test)
        
        #make it optional to show plots to console
        if show_plots: 
            print("\n")

        if has_test:
            return (grouped, grouped_test)
        else:
            return grouped


def get_univariate_plots( data, target_col,features_list=0, bins=10, data_test=0,show_plots=True,export_to_excel=False):
    
    """
        Creates univariate dependence plots for features in the dataset
        :param data: dataframe containing features and target columns
        :param target_col: target column name
        :param features_list: by default creates plots for all features. If 
        list passed, creates plots of only those features
        :param bins: number of bins to be created from continuous feature
        :param data_test: test data which has to be compared with input 
        data for correlation
        :param show_plots: if to show plots on the screen (stdout) - they can be so many , 1000 features
        :param export_to_excel: if to generate excel report with plots instead
        :return: Draws univariate plots for all columns in data
    """

    #init plots to excel object at entry
    xlsx = None
    if export_to_excel:
        xlsx = ExportPlotsToExcel()

    #make it optional to show plots to console
    #if false use matplotlib bancground engine
    if show_plots==False:
        matplotlib.use('Agg')
   
    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    has_test = type(data_test) == pd.core.frame.DataFrame

    
    for i in range(len(features_list)):
        cols = features_list[i]
        if cols != target_col and data[cols].dtype == "O":
            print(
                cols
                + " is categorical. Categorical features not supported yet."
            )
        elif cols != target_col and data[cols].dtype != "O":
            univariate_plotter(
                feature=cols,
                data=data,
                target_col=target_col,
                bins=bins,
                data_test=data_test,
                plot_number=i+1,
                show_plots=show_plots,
                export_to_excel=export_to_excel,
                xlsx=xlsx
                
            )
    

    #if export to excel true, add stats sheet to the workbook
    if export_to_excel:
        if has_test:
            stats = get_trend_stats(data=data, target_col=target_col, data_test=data_test)
            stats.to_excel(xlsx.writer, sheet_name='STATS',index=False)
            xlsx.writer.save()
        else:
            stats = get_trend_stats(data=data, target_col=target_col)
            stats.to_excel(xlsx.writer, sheet_name='STATS',index=False)
            xlsx.writer.save()


        from IPython.display import FileLink,display
        display(FileLink(xlsx.path))

        os.system("rm *.png")
    


def get_trend_stats(data, target_col, features_list=0, bins=10, data_test=0):
    """
        Calculates trend changes and correlation between train/test for 
        list of features.
        
        :param data: dataframe containing features and target columns.
        :param target_col: target column name.
        :param features_list: by default creates plots for all features. If 
        list passed, creates plots of only those features.
        :param bins: number of bins to be created from continuous feature.
        :param data_test: test data which has to be compared with input data 
        for correlation.
        :return: dataframe with trend changes and trend correlation 
        (if test data passed).
    """

    if type(features_list) == int:
        features_list = list(data.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(data_test) == pd.core.frame.DataFrame
    ignored = []
    for feature in features_list:
        if data[feature].dtype == "O" or feature == target_col:
            ignored.append(feature)
        else:
            cuts, grouped = get_grouped_data(
                input_data=data,
                feature=feature,
                target_col=target_col,
                bins=bins,
            )
            trend_changes = get_trend_changes(
                grouped_data=grouped, feature=feature, target_col=target_col
            )
            if has_test:
                grouped_test = get_grouped_data(
                    input_data=data_test.reset_index(drop=True),
                    feature=feature,
                    target_col=target_col,
                    bins=bins,
                    cuts=cuts,
                )
                trend_corr = get_trend_correlation(
                    grouped, grouped_test, feature, target_col
                )
                trend_changes_test = get_trend_changes(
                    grouped_data=grouped_test,
                    feature=feature,
                    target_col=target_col,
                )
                stats = [
                    feature,
                    trend_changes,
                    trend_changes_test,
                    trend_corr,
                ]
            else:
                stats = [feature, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = (
        ["Feature", "Trend_changes"]
        if has_test == False
        else [
            "Feature",
            "Trend_changes",
            "Trend_changes_test",
            "Trend_correlation",
        ]
    )
    if len(ignored) > 0:
        print(
            "Categorical features "
            + str(ignored)
            + " ignored. Categorical features not supported yet."
        )

    print("Returning stats for all numeric features")
    return stats_all_df
