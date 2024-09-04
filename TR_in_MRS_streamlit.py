import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from matplotlib import MatplotlibDeprecationWarning
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")


# Data upload
datasets = {
    'V1': {
        'TR2_064': pd.read_csv('Interactive_TRISP_data/V1_TR2_064.csv'),
        'TR2_128': pd.read_csv('Interactive_TRISP_data/V1_TR2_128.csv'),
        'TR5_064': pd.read_csv('Interactive_TRISP_data/V1_TR5_064.csv'),
        'TR8_032': pd.read_csv('Interactive_TRISP_data/V1_TR8_032.csv'),
        'TR8_064': pd.read_csv('Interactive_TRISP_data/V1_TR8_064.csv')
    },
    'V2': {
        'TR2_064': pd.read_csv('Interactive_TRISP_data/V2_TR2_064.csv'),
        'TR2_128': pd.read_csv('Interactive_TRISP_data/V2_TR2_128.csv'),
        'TR5_064': pd.read_csv('Interactive_TRISP_data/V2_TR5_064.csv'),
        'TR8_032': pd.read_csv('Interactive_TRISP_data/V2_TR8_032.csv'),
        'TR8_064': pd.read_csv('Interactive_TRISP_data/V2_TR8_064.csv')
    },
    'V3': {
        'TR2_064': pd.read_csv('Interactive_TRISP_data/V3_TR2_064.csv'),
        'TR2_128': pd.read_csv('Interactive_TRISP_data/V3_TR2_128.csv'),
        'TR5_064': pd.read_csv('Interactive_TRISP_data/V3_TR5_064.csv'),
        'TR8_032': pd.read_csv('Interactive_TRISP_data/V3_TR8_032.csv'),
        'TR8_064': pd.read_csv('Interactive_TRISP_data/V3_TR8_064.csv')
    },
    'V4': {
        'TR2_064': pd.read_csv('Interactive_TRISP_data/V4_TR2_064.csv'),
        'TR2_128': pd.read_csv('Interactive_TRISP_data/V4_TR2_128.csv'),
        'TR5_064': pd.read_csv('Interactive_TRISP_data/V4_TR5_064.csv'),
        'TR8_032': pd.read_csv('Interactive_TRISP_data/V4_TR8_032.csv'),
        'TR8_064': pd.read_csv('Interactive_TRISP_data/V4_TR8_064.csv')
    },
    'V5': {
        'TR2_064': pd.read_csv('Interactive_TRISP_data/V5_TR2_064.csv'),
        'TR2_128': pd.read_csv('Interactive_TRISP_data/V5_TR2_128.csv'),
        'TR5_064': pd.read_csv('Interactive_TRISP_data/V5_TR5_064.csv'),
        'TR8_032': pd.read_csv('Interactive_TRISP_data/V5_TR8_032.csv'),
        'TR8_064': pd.read_csv('Interactive_TRISP_data/V5_TR8_064.csv')
    },
}

# Define a mapping from current TR labels to desired labels
tr_mapping = {
    'TR2_128': 'TR=2s, 128acqs',
    'TR2_064': 'TR=2s, 64acqs',
    'TR5_064': 'TR=5s, 64acqs',
    'TR8_064': 'TR=8s, 64acqs',
    'TR8_032': 'TR=8s, 32acqs',
}

trs = [tr_mapping[tr] for tr in datasets['V1'].keys()]

############################ Introduction ###########################
def page_intro():
    st.markdown("""# An Investigation of the Impact of Repetition Time on MR Spectroscopy""")
    st.markdown("""## Purpose:""")
    st.markdown("""#### This study aims to determine the optimal balance between scan time and repetition time that minimizes T$_1$ weighting effects.""")
    st.markdown("""## Study Details:""")
    st.markdown("""- 5 healthy controls (mean age 25 $\pm$ 2 years)""")
    st.markdown("""- 3T Philips Ingenia Elition X""")
    st.markdown("""- Semi-LASER localization""")
    st.markdown("""- 30 x 20 x 13 mm$^3$ voxel in the posterior cingulate cortex (PCC)""")
    st.markdown("""## How to Use This Tool:""")
    st.markdown("""- This document reflects the data taken in this study.""")
    st.markdown("""- Each page reflects various aspects of analysis conducted in this study.""")
    st.markdown("""- Navigate to various analysis pages via the menu on the left.""")
    st.markdown("""- Upon selecting a page, the default selection of data is what is presented in the manuscript.""")
    st.markdown("""- Modify the selections by choosing different items in the drop-down menus, such as the choice of metabolite or the number of volunteers included in the plot.""")

    
######################## Data representation ########################
def vol_dat():
    st.header("MRS Data Representation")

    st.markdown("""#### Explanation:""")
    st.markdown("""In this section, you can navigate through the data used for the entire study. The spectra for each volunteer at each TR and acquisition are included.""")
    st.markdown("""The extracted data from the fit of that dataset is presented in the table below.""")


    st.markdown("""### Select the volunteer and TR of your choice:""")
    
    #Creates list of volunteers
    vols= list(datasets.keys())
    #Creates list of TRs
    #trs = list(datasets['V1'].keys())

    # Create a select box for the datasets
    selected_V_spec = st.selectbox('Select a volunteer:', vols, key='spec_v')
    selected_tr_label = st.selectbox('Select a TR:', trs)
    selected_tr = list(tr_mapping.keys())[list(tr_mapping.values()).index(selected_tr_label)]

    image_path = f'Interactive_TRISP_data/figs/{selected_V_spec}_{selected_tr}.png'

    st.markdown(f"""### Spectrum with fit for volunteer {selected_V_spec}, {selected_tr_label}""")
    st.image(image_path)

    st.markdown(f"""### Key data table for volunteer {selected_V_spec}, {selected_tr_label}""")
    
    # Display the selected dataset as a table
    selected_dataset = datasets[selected_V_spec][selected_tr]
    Cols_to_display = ['Metab','SNR','mM','mM CRLB', 'FWHM'];
    selected_dataset=selected_dataset[Cols_to_display]
    #rows_to_drop = []
    rows_to_drop = [11, 12, 13, 14, 15, 16, 17]
    selected_dataset = selected_dataset.drop(rows_to_drop)
    selected_dataset=selected_dataset.T
    # Format numeric values to 4 decimal places
    selected_dataset = selected_dataset.map(lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x)

    st.dataframe(selected_dataset)
    
############################ SNR content ############################
def page_snr():
    st.header("Signal to Noise Ratio (SNR)")

    st.markdown("""#### Explanation:""")
    st.markdown("""This section accompanies Figure 2 of the manuscript.""")
    st.markdown("""Here, you can see how different TRs and the number of acquisitions affect the SNR.""")

    st.markdown("""- The first row of figures represents the absolute SNR for the chosen metabolite. The second row shows the same data, normalized to a TR of 8 seconds.""")
    st.markdown("""- The first column of figures represents when scan time was kept as similar as possible, while the second column keeps the number of acquisitions the same.""")


    st.markdown("""### Select the volunteer and metabolite of your choice:""")
                
    # Assuming the first column contains the metabolite names for all sets of data
    all_metabolites = list(datasets['V1']['TR2_064'].iloc[:,0].unique()) 

    # List of metabolites to exclude
    exclude_metabolites = ['Cr+PCr', 'Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30']

    # Filter out the excluded metabolites
    metabolites = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]
    
    # Create a multiselect box for the volunteers for plot with an "All data sets" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'All data sets')
    selected_Vs_plot = st.multiselect('Select a volunteer:', v_options, ['All data sets'], key='plot_v')

    # If "All data sets" is selected, select all the V datasets
    if 'All data sets' in selected_Vs_plot:
        selected_Vs_plot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    selected_metabolite_plot = st.selectbox('Select a metabolite:', metabolites, key='plot_metabolite', index=metabolites.index('NAA') if 'NAA' in metabolites else 0)

    # Define the TR datasets to plot
    tr_datasets_1 = ['TR2_128', 'TR5_064', 'TR8_032']
    tr_datasets_2 = ['TR2_064', 'TR5_064', 'TR8_064']
    tr_datasets_3 = ['TR2_128', 'TR5_064', 'TR8_032']
    tr_datasets_4 = ['TR2_064', 'TR5_064', 'TR8_064']

    # Define the corresponding TR values
    tr_values = [2, 5, 8]

    # Define the colors for each volunteer
    #colors = {
    #    'V1': '#541747', #Palatinate - Purple
    #    'V2': '#3B6A4A', #Hunter Green
    #    'V3': '#4DA167', #Shamrock Green
    #    'V4': '#3BC14A', #Dark pastel green
    #    'V5': '#D499B9'  #Lilac
    #}

    # Define the colors for each volunteer
    #colors = {
    #    'V1': '#39A2AE', #Moonstone - light blue
    #    'V2': '#55DBCB', #Turquoise
    #    'V3': '#75E4B3', #Aquamarine
    #    'V4': '#963484', #Plum
    #    'V5': '#EF798A'  #Bright Pink
    #}

    # Define the colors for each volunteer
    colors = {
        'V1': '#EC1313', #Red (CMYK)
        'V2': '#EC7F11', #Tangerine - Orange
        'V3': '#0DBE1E', #Dark pastel green
        'V4': '#151CDD', #Chrysler blue
        'V5': '#7B065C'  #Byzantium - Purple
    }

    marker_shapes = {
        'V1': 'P',
        'V2': 'D',
        'V3': '^',
        'V4': 's',
        'V5': 'X'
    }
    
    
    # Create a dictionary that maps the original labels to the new labels
    label_map = {'GPC+PCh': 'tCho', 'Cr+PCr': 'tCr'}
    
    # Create the subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    fig.subplots_adjust(hspace=0.4)

    # Initialize lists to store the SNR values for all volunteers
    all_snr_values_1 = []
    all_snr_values_2 = []
    all_snr_values_3 = []
    all_snr_values_4 = []

    # Loop over the selected V datasets
    for selected_V_plot in selected_Vs_plot:
        snr_values_1 = []
        snr_values_2 = []
        snr_values_3 = []
        snr_values_4 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2, tr_dataset_3, tr_dataset_4 in zip(tr_datasets_1, tr_datasets_2, tr_datasets_3, tr_datasets_4):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_plot][tr_dataset_1]
            df_2 = datasets[selected_V_plot][tr_dataset_2]
            df_3 = datasets[selected_V_plot][tr_dataset_3]
            df_4 = datasets[selected_V_plot][tr_dataset_4]

            # Filter the DataFrame based on the selected metabolite
            filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_3 = df_3[df_3.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_4 = df_4[df_4.iloc[:, 0] == selected_metabolite_plot]

            # Get the SNR value and append it to the list
            if not filtered_df_plot_1.empty:
                snr_value_1 = filtered_df_plot_1['SNR'].values[0]
                snr_values_1.append(snr_value_1)
            if not filtered_df_plot_2.empty:
                snr_value_2 = filtered_df_plot_2['SNR'].values[0]
                snr_values_2.append(snr_value_2)
            if not filtered_df_plot_3.empty:
                snr_value_3 = filtered_df_plot_3['SNR'].values[0]
                snr_values_3.append(snr_value_3)
            if not filtered_df_plot_4.empty:
                snr_value_4 = filtered_df_plot_4['SNR'].values[0]
                snr_values_4.append(snr_value_4)

        # Normalize the SNR values by the last value in the list
        if snr_values_3:
            snr_values_3 = [value / snr_values_3[-1] for value in snr_values_3]
        if snr_values_4:
            snr_values_4 = [value / snr_values_4[-1] for value in snr_values_4]

        # Add the SNR values to the lists for all volunteers
        if snr_values_1:
            all_snr_values_1.append(snr_values_1)
        if snr_values_2:
            all_snr_values_2.append(snr_values_2)
        if snr_values_3:
            all_snr_values_3.append(snr_values_3)
        if snr_values_4:
            all_snr_values_4.append(snr_values_4)

        # Plot the SNR values for the current V dataset
        axs[0, 0].plot(tr_values, snr_values_1, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)
        axs[0, 1].plot(tr_values, snr_values_2, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)
        axs[1, 0].plot(tr_values, snr_values_3, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)
        axs[1, 1].plot(tr_values, snr_values_4, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)

    # Calculate the average SNR values for all volunteers
    if all_snr_values_3:
        avg_snr_values_3 = np.mean(all_snr_values_3, axis=0)
        axs[1, 0].plot(tr_values, avg_snr_values_3, marker='.', linestyle='--', label='Average', color='black', linewidth=4)
    if all_snr_values_4:
        avg_snr_values_4 = np.mean(all_snr_values_4, axis=0)
        axs[1, 1].plot(tr_values, avg_snr_values_4, marker='.', linestyle='--', label='Average', color='black', linewidth=4)


    global_fontsize = 16;

    axs[0, 0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0, 0].set_ylabel('SNR', fontsize=global_fontsize)
    axs[0, 0].set_title(f'Similar scan time, SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[0, 0].grid(True)
    axs[0, 0].set_xticks([2, 5, 8])
    axs[0, 0].tick_params(axis='both', labelsize=global_fontsize);

    axs[0, 1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0, 1].set_ylabel('SNR', fontsize=global_fontsize)
    axs[0, 1].set_title(f'Same number of acquisitions, SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks([2, 5, 8])
    axs[0, 1].tick_params(axis='both', labelsize=global_fontsize);

    axs[1, 0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1, 0].set_ylabel('Normalized SNR', fontsize=global_fontsize)
    axs[1, 0].set_title(f'Similar scan time, normalized SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks([2, 5, 8])
    axs[1, 0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1, 1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1, 1].set_ylabel('Normalized SNR', fontsize=global_fontsize)
    axs[1, 1].set_title(f'Same number of acquisitions, normalized SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[1, 1].grid(True)
    axs[1, 1].set_xticks([2, 5, 8])
    axs[1, 1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_plot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0, 0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0, 0].legend(handles1s, labels1s, fontsize = 13)

        # Get the handles and labels for the third subplot
        handles3, labels3 = axs[1, 0].get_legend_handles_labels()

        # Sort them by labels
        labels3s, handles3s = zip(*sorted(zip(labels3, handles3), key=lambda t: t[0]))

        # Set the legend
        axs[1, 0].legend(handles3s, labels3s, fontsize = 13)
    #    axs[0, 0].legend() 
    #    axs[1, 0].legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

######################### SNR per unit time #########################
def page_snr_per_time():
    st.header("SNR per Unit Scan Time")

    st.markdown("""#### Explanation:""")
    st.markdown("""This page accompanies Figure 3 in the manuscript.""")
    st.markdown("""The SNR for a metabolite is divided by the scan time at each TR and is plotted for every volunteer. The average of all volunteers is represented by a solid black line.""")
    st.markdown("""The calculated SNR is shown as a dashed red line. This calculation is based on the expected increase or decrease in SNR when moving from one TR to another, due to additional acquisitions in a given time period. You can change the reference point, and the calculated values will adjust accordingly.""")

    st.markdown("""### Select the metabolite and reference TR of your choice:""")

    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    # Define your times
    TR2_time = 264/60
    TR5_time = 340/60
    TR8_time = 544/60

    # Get a list of all metabolites
    all_metabolites = list(datasets['V1']['TR2_064'].iloc[:,0].unique()) 

    # List of metabolites to exclude
    exclude_metabolites = ['Cr+PCr', 'Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30']

    # Filter out the excluded metabolites
    metabolites = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]

    # Let the user select the metabolite
    selected_metabolite = st.selectbox('Select a metabolite:', metabolites, index=metabolites.index('NAA'))

    # Allow the user to select a "reference" TR
    tr_pt_mapping = {
    'TR2_128': 'TR=2s, 128acqs',
    'TR5_064': 'TR=5s, 64acqs',
    'TR8_064': 'TR=8s, 64acqs',
    }

    trs_pt = list(tr_pt_mapping.values())

    reference_TR_label = st.selectbox('Select a reference TR:', trs_pt, index=2)
    reference_TR = [key for key, value in tr_pt_mapping.items() if value == reference_TR_label][0]

    # Calculate SNR per time for each volunteer and each TR
    SNR_per_time = {}
    for volunteer in datasets:
        SNR_per_time[volunteer] = {}
        for TR in datasets[volunteer]:
            time = TR2_time if 'TR2' in TR else TR5_time if 'TR5' in TR else TR8_time
            metabolite_num = datasets[volunteer][TR].index[datasets[volunteer][TR].iloc[:, 0] == selected_metabolite][0]
            SNR_per_time[volunteer][TR] = datasets[volunteer][TR].at[metabolite_num, "SNR"] / time




    # Create a DataFrame for plotting
    SNR_per_time_df = pd.DataFrame({
        'Category': ['TR2_128'] * len(datasets) + ['TR5_064'] * len(datasets) + ['TR8_064'] * len(datasets),
        'Value': [SNR_per_time[v]['TR2_128'] for v in SNR_per_time] +
                [SNR_per_time[v]['TR5_064'] for v in SNR_per_time] +
                [SNR_per_time[v]['TR8_064'] for v in SNR_per_time]
    })

    # Determine the mean value for the selected TR SNR per time
    mean_SNR_per_time = SNR_per_time_df[SNR_per_time_df['Category'] == reference_TR]['Value'].mean()

    # Calculate the values of the red lines based on the selected reference TR
    line_values = {
        'TR2_128': mean_SNR_per_time * np.sqrt(2/2),
        'TR5_064': mean_SNR_per_time * np.sqrt(2/5),
        'TR8_064': mean_SNR_per_time * np.sqrt(2/8)
    } if reference_TR == 'TR2_128' else {
        'TR2_128': mean_SNR_per_time * np.sqrt(5/2),
        'TR5_064': mean_SNR_per_time * np.sqrt(5/5),
        'TR8_064': mean_SNR_per_time * np.sqrt(5/8)
    } if reference_TR == 'TR5_064' else {
        'TR2_128': mean_SNR_per_time * np.sqrt(8/2),
        'TR5_064': mean_SNR_per_time * np.sqrt(8/5),
        'TR8_064': mean_SNR_per_time * np.sqrt(8/8)
    }

    # Plot the data
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.stripplot(data=SNR_per_time_df, x='Category', y='Value', jitter=False, alpha=0.7, s=25, color='#0c2343', ax=ax)
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 6},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Category",
                y="Value",
                data=SNR_per_time_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

    # Add a horizontal line at y=10 for TR2, y=8 for TR5, and y=6 for TR8
    for i, (_, line_value) in enumerate(line_values.items()):
        ax.plot([i-0.4, i+0.4], [line_value, line_value], color='r', linestyle='--', linewidth=6, zorder=11)


    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Volunteers',
                            markerfacecolor='#0c2343', markersize=25, alpha=0.7),
                    Line2D([0], [0], color='k', lw=6, label='Average'),
                    Line2D([0], [0], color='r', lw=6, linestyle='--', label='Calculated')]
    ax.legend(handles=legend_elements, fontsize=20)
    
    # Set x-axis labels to be 2, 5, and 8
    ax.set_xticks(range(len(line_values)))
    ax.set_xticklabels(['2', '5', '8'])

    plt.xlabel("Repetition time, TR [s]", fontsize=20)
    plt.ylabel("SNR per unit scan time (SNR/minute)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Display the plot in Streamlit
    st.pyplot(fig)

############################ Concentration ##########################
def page_concentration():
    st.header("Concentration")

    st.markdown("""#### Explanation:""")
    st.markdown("""- This page accompanies Figure 4 in the manuscript.""")
    st.markdown("""- Here, you can see how different TRs and the number of acquisitions affect the reported concentration.""")
    st.markdown("""- In the first row of figures, the concentration of one metabolite is plotted for all volunteers. The figure on the left represents acquisitions with a similar scan time, and the figure on the right represents a constant number of acquisitions.""")
    st.markdown("""- In the second row of figures, only the average trend is plotted for each metabolite. However, multiple metabolites can be selected.""")

    st.markdown("""## Single metabolite analysis""")
    st.markdown("""### Select the volunteer and metabolite of your choice:""")
    
    # Assuming the first column contains the metabolite names for all sets of data
    all_metabolites = list(datasets['V1']['TR2_064'].iloc[:,0].unique()) 

    # List of metabolites to exclude
    exclude_metabolites = ['Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30', 'Cr', 'PCr', 'GPC', 'PCh']

    # Filter out the excluded metabolites
    metabolites_conc = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]

    # Create a multiselect box for the volunteers for plot with an "All data sets" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'All data sets')
    selected_Vs_conc = st.multiselect('Select a volunteer:', v_options, ['All data sets'], key='plot_v')  # Set 'All data sets' as the default option

    # If "All data sets" is selected, select all the V datasets
    if 'All data sets' in selected_Vs_conc:
        selected_Vs_conc = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    # metabolites_conc_avg = list(datasets[selected_V_conc][selected_TR_conc].iloc[:,0].unique())

    # Add the combined metabolite to the list
    metabolites_conc.append('GPC+PCh')

    selected_metabolite_plot = st.selectbox('Select a metabolite:', metabolites_conc, key='plot_metabolite', index=metabolites_conc.index('NAA') if 'NAA' in metabolites_conc else 0)

    label_map = {'GPC+PCh': 'tCho', 'Cr+PCr': 'tCr'}

    # Define the corresponding TR values
    tr_values = [2, 5, 8]

    # Define the TR datasets to plot
    tr_datasets_1 = ['TR2_128', 'TR5_064', 'TR8_032']
    tr_datasets_2 = ['TR2_064', 'TR5_064', 'TR8_064']

    
    # Define the colors for each volunteer
    colors = {
        'V1': '#EC1313', #Red (CMYK)
        'V2': '#EC7F11', #Tangerine - Orange
        'V3': '#0DBE1E', #Dark pastel green
        'V4': '#151CDD', #Chrysler blue
        'V5': '#7B065C'  #Byzantium - Purple
    }

    marker_shapes = {
        'V1': 'P',
        'V2': 'D',
        'V3': '^',
        'V4': 's',
        'V5': 'X'
    }


   
    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(top=1.2)

    # Initialize lists to store the SNR values for all volunteers
    all_conc_values_1 = []
    all_conc_values_2 = []

    # Loop over the selected V datasets
    for selected_V_conc in selected_Vs_conc:
        conc_values_1 = []
        conc_values_2 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_conc][tr_dataset_1]
            df_2 = datasets[selected_V_conc][tr_dataset_2]

            # Check if the selected metabolite is the combined one
            if selected_metabolite_plot == 'GPC+PCh':
                metabolites_to_combine = ['GPC', 'PCh']
            else:
                metabolites_to_combine = [selected_metabolite_plot]

            # Initialize the combined conc values
            combined_conc_value_1 = 0
            combined_conc_value_2 = 0

            # Loop over the metabolites to combine
            for metabolite_to_combine in metabolites_to_combine:
                # Filter the DataFrame based on the current metabolite
                filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == metabolite_to_combine]
                filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == metabolite_to_combine]

                # Get the conc value and add it to the combined conc value
                if not filtered_df_plot_1.empty:
                    conc_value_1 = filtered_df_plot_1['mM'].values[0]
                    combined_conc_value_1 += conc_value_1
                if not filtered_df_plot_2.empty:
                    conc_value_2 = filtered_df_plot_2['mM'].values[0]
                    combined_conc_value_2 += conc_value_2

            # Append the combined conc values to the lists
            if combined_conc_value_1:
                conc_values_1.append(combined_conc_value_1)
            if combined_conc_value_2:
                conc_values_2.append(combined_conc_value_2)

        # Add the conc values to the lists for all volunteers
        if conc_values_1:
            all_conc_values_1.append(conc_values_1)
        if conc_values_2:
            all_conc_values_2.append(conc_values_2)

        # Plot the conc values for the current V dataset
        axs[0].plot(tr_values, conc_values_1, marker=marker_shapes[selected_V_conc], label=f'Volunteer {selected_V_conc[1]}', color=colors[selected_V_conc], linewidth=4, markersize = 10)
        axs[1].plot(tr_values, conc_values_2, marker=marker_shapes[selected_V_conc], label=f'Volunteer {selected_V_conc[1]}', color=colors[selected_V_conc], linewidth=4, markersize = 10)

    # Calculate the average conc values for all volunteers
    if all_conc_values_1:
        avg_conc_values_1 = np.mean(all_conc_values_1, axis=0)
        axs[0].plot(tr_values, avg_conc_values_1, marker='.', linestyle='--', label=f'Average {selected_metabolite_plot}', color='black', linewidth=4)
    if all_conc_values_2:
        avg_conc_values_2 = np.mean(all_conc_values_2, axis=0)
        axs[1].plot(tr_values, avg_conc_values_2, marker='.', linestyle='--', label=f'Average {selected_metabolite_plot}', color='black', linewidth=4)

    global_fontsize = 16

    axs[0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0].set_ylabel('Apparent concentration (mM)', fontsize=global_fontsize)
    axs[0].set_title(f'Similar scan time, reported conc. comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0].grid(True)
    axs[0].set_xticks([2, 5, 8])
    axs[0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1].set_ylabel('Apparent concentration (mM)', fontsize=global_fontsize)
    axs[1].set_title(f'Same number of acqs., reported conc. comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[1].grid(True)
    #axs[1].set_facecolor('black')
    axs[1].set_xticks([2, 5, 8])
    axs[1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_conc:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0].legend(handles1s, labels1s, fontsize = 13)

    # Display the plot in Streamlit
    st.pyplot(fig)

    ###### Plot the average of the metabolite for many metabolites #########

    st.markdown("""## Many metabolite analysis""")
    st.markdown("""### Select your choice of metabolites:""")

    # Create the subplots
    fig2, axs2 = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(top=1.2)


    # Assuming the first column contains the metabolite names for all sets of data
    all_metabolites = list(datasets['V1']['TR2_064'].iloc[:,0].unique()) 

    # List of metabolites to exclude
    exclude_metabolites = ['Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30']

    # Filter out the excluded metabolites
    metabolites_conc_avg = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]


    # # Assuming the first column contains the metabolite names for all sets of data
    # metabolites_conc_avg = list(datasets[selected_V_conc]['TR2_128'].iloc[:,0].unique())

    # Add the combined metabolite to the list
    metabolites_conc_avg.append('GPC+PCh')

    selected_metabolites_plot = st.multiselect('Select metabolites:', metabolites_conc_avg, ['NAA', 'Cr+PCr', 'GPC+PCh', 'mI', 'Glu'], key='average_plot_metabolite')


    # Create a color map
    cmap = plt.get_cmap('jet')

    # Create a dictionary to map each metabolite to a color
    color_dict = {}
    for i, metabolite in enumerate(metabolites_conc_avg):
        color_dict[metabolite] = cmap(i / len(metabolites_conc_avg))

    # Always calculate the average for all 5 volunteers
    all_Vs = list(datasets.keys())

    # Loop over the selected metabolites
    for selected_metabolite_plot in selected_metabolites_plot:
        # Initialize lists to store the conc values for all volunteers
        all_conc_values_1 = []
        all_conc_values_2 = []

        # Check if the selected metabolite is the combined one
        if selected_metabolite_plot == 'GPC+PCh':
            metabolites_to_combine = ['GPC', 'PCh']
        else:
            metabolites_to_combine = [selected_metabolite_plot]

        # Loop over all the V datasets
        for selected_V_conc in all_Vs:
            conc_values_1 = []
            conc_values_2 = []
            # Loop over the TR datasets
            for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
                # Initialize the combined conc values
                combined_conc_value_1 = 0
                combined_conc_value_2 = 0
                # Loop over the metabolites to combine
                for metabolite_to_combine in metabolites_to_combine:
                    # Get the selected DataFrame
                    df_1 = datasets[selected_V_conc][tr_dataset_1]
                    df_2 = datasets[selected_V_conc][tr_dataset_2]

                    # Filter the DataFrame based on the current metabolite
                    filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == metabolite_to_combine]
                    filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == metabolite_to_combine]

                    # Get the conc value and add it to the combined conc value
                    if not filtered_df_plot_1.empty:
                        conc_value_1 = filtered_df_plot_1['mM'].values[0]
                        combined_conc_value_1 += conc_value_1
                    if not filtered_df_plot_2.empty:
                        conc_value_2 = filtered_df_plot_2['mM'].values[0]
                        combined_conc_value_2 += conc_value_2

                # Append the combined conc values to the lists
                if combined_conc_value_1:
                    conc_values_1.append(combined_conc_value_1)
                if combined_conc_value_2:
                    conc_values_2.append(combined_conc_value_2)

            # Add the conc values to the lists for all volunteers
            if conc_values_1:
                all_conc_values_1.append(conc_values_1)
            if conc_values_2:
                all_conc_values_2.append(conc_values_2)

        # Calculate the average conc values for all volunteers
        if all_conc_values_1:
            avg_conc_values_1 = np.mean(all_conc_values_1, axis=0)
            # Normalize the average values by the average value at TR8
            avg_conc_values_1 = avg_conc_values_1 / avg_conc_values_1[-1]
            axs2[0].plot(tr_values, avg_conc_values_1, marker='o', linestyle='-', label=f'{selected_metabolite_plot}', color=color_dict[selected_metabolite_plot], linewidth=5, markersize=12)
            axs2[0].set_facecolor('black')
            axs2[0].set_xlabel('TR (s)', fontsize=global_fontsize)
            axs2[0].set_ylabel('Normalized apparent concentration', fontsize=global_fontsize)
            axs2[0].set_title(f'Similar scan time, normalized reported conc.', fontsize=global_fontsize+4)
            axs2[0].grid(True)
            axs2[0].set_xticks([2, 5, 8])
            axs2[0].tick_params(axis='both', labelsize=global_fontsize);
        if all_conc_values_2:
            avg_conc_values_2 = np.mean(all_conc_values_2, axis=0)
            # Normalize the average values by the average value at TR8
            avg_conc_values_2 = avg_conc_values_2 / avg_conc_values_2[-1]
            axs2[1].plot(tr_values, avg_conc_values_2, marker='o', linestyle='-', label=f'{selected_metabolite_plot}', color=color_dict[selected_metabolite_plot], linewidth=5, markersize=12)
            axs2[1].set_facecolor('black')
            axs2[1].set_xlabel('TR (s)', fontsize=global_fontsize)
            axs2[1].set_ylabel('Normalized apparent concentration', fontsize=global_fontsize)
            axs2[1].set_title(f'Same number of acqs., normalized reported conc.', fontsize=global_fontsize+4)
            axs2[1].grid(True)
            axs2[1].set_xticks([2, 5, 8])
            axs2[1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one metabolite is selected
    if selected_metabolites_plot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs2[0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs2[0].legend(handles1s, labels1s, fontsize = 13)

    # Display the plot in Streamlit
    st.pyplot(fig2)

############################### CRLB ################################
def page_CRLB():
    st.header("Cramér-Rao Lower Bound (CRLB)")

    st.markdown("""#### Explanation:""")
    st.markdown("""- Here, you can see how the CRLBs change with different TRs and the number of acquisitions.""")
    st.markdown("""- In the first section, we have a similar comparison to other sections where we compare the CRLBs at different TRs. The left column represents data acquired with similar scan times, and the second column represents the same number of acquisitions. The first row represents absolute CRLB values in units of mM, while the second row represents relative CRLB values with respect to the metabolite concentration, in units of percent.""")
    st.markdown("""- In the second section, the concentrations of all experiments for one metabolite are compared to the absolute CRLB values of all experiments for the same metabolite. If more than 20% of the CRLB values exceed the threshold, then that metabolite is considered to be poorly fit. The threshold is determined to be 30% of the median value of the concentration values.""")
    st.markdown("""- It was this CRLB comparison that dictated which metabolites are permitted to be plotted throughout this app.""")

    st.markdown("""## CRLB vs TR analysis""")
    st.markdown("""### Select the volunteer and metabolite of your choice:""")


    # Create a multiselect box for the volunteers for plot with an "All data sets" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'All data sets')
    selected_Vs_CRLBplot = st.multiselect('Select a volunteer:', v_options, ['All data sets'], key='plot_CRLB_v')

    # If "All data sets" is selected, select all the V datasets
    if 'All data sets' in selected_Vs_CRLBplot:
        selected_Vs_CRLBplot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    all_metabolites = list(datasets['V1']['TR2_064'].iloc[:,0].unique())

    # List of metabolites to exclude
    exclude_metabolites = ['Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30', 'Cr', 'PCr']

    # Filter out the excluded metabolites
    metabolites_CRLB = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]

    # Add the combined metabolite to the list
    #metabolites_CRLB.append('GPC+PCh')

    selected_metabolite_plot = st.selectbox('Select a metabolite:', metabolites_CRLB, key='plot_metabolite', index=metabolites_CRLB.index('NAA') if 'NAA' in metabolites_CRLB else 0)

    # Define the corresponding TR values
    tr_values = [2, 5, 8]

    # Define the TR datasets to plot
    tr_datasets_1 = ['TR2_128', 'TR5_064', 'TR8_032']
    tr_datasets_2 = ['TR2_064', 'TR5_064', 'TR8_064']

    # Define the colors for each volunteer
    colors = {
        'V1': '#EC1313', #Red (CMYK)
        'V2': '#EC7F11', #Tangerine - Orange
        'V3': '#0DBE1E', #Dark pastel green
        'V4': '#151CDD', #Chrysler blue
        'V5': '#7B065C'  #Byzantium - Purple
    }

    marker_shapes = {
        'V1': 'P',
        'V2': 'D',
        'V3': '^',
        'V4': 's',
        'V5': 'X'
    }

    label_map = {'Cr+PCr': 'tCr'}

    # Create the subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    plt.subplots_adjust(top=1.2)

    # Initialize lists to store the CRLB values for all volunteers
    all_CRLB_values_1 = []
    all_CRLB_values_2 = []
    all_percent_CRLB_values_1 = []
    all_percent_CRLB_values_2 = []

    # Loop over the selected V datasets
    for selected_V_CRLBplot in selected_Vs_CRLBplot:
        CRLB_values_1 = []
        CRLB_values_2 = []
        percent_CRLB_values_1 = []
        percent_CRLB_values_2 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_CRLBplot][tr_dataset_1]
            df_2 = datasets[selected_V_CRLBplot][tr_dataset_2]

            # Initialize the combined CRLB values
            combined_CRLB_value_1 = 0
            combined_CRLB_value_2 = 0
            combined_percent_CRLB_value_1 = 0
            combined_percent_CRLB_value_2 = 0

            # Filter the DataFrame based on the current metabolite
            filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == selected_metabolite_plot]

            # Get the CRLB value and add it to the combined CRLB value
            if not filtered_df_plot_1.empty:
                CRLB_value_1 = filtered_df_plot_1['mM CRLB'].values[0]
                percent_CRLB_value_1 = filtered_df_plot_1['%CRLB'].values[0]
                combined_CRLB_value_1 += CRLB_value_1
                combined_percent_CRLB_value_1 += percent_CRLB_value_1
            if not filtered_df_plot_2.empty:
                CRLB_value_2 = filtered_df_plot_2['mM CRLB'].values[0]
                percent_CRLB_value_2 = filtered_df_plot_2['%CRLB'].values[0]
                combined_CRLB_value_2 += CRLB_value_2
                combined_percent_CRLB_value_2 += percent_CRLB_value_2

            # Append the combined CRLB values to the lists
            if combined_CRLB_value_1:
                CRLB_values_1.append(combined_CRLB_value_1)
                percent_CRLB_values_1.append(combined_percent_CRLB_value_1)
            if combined_CRLB_value_2:
                CRLB_values_2.append(combined_CRLB_value_2)
                percent_CRLB_values_2.append(combined_percent_CRLB_value_2)

        # Add the CRLB values to the lists for all volunteers
        if CRLB_values_1:
            all_CRLB_values_1.append(CRLB_values_1)
            all_percent_CRLB_values_1.append(percent_CRLB_values_1)
        if CRLB_values_2:
            all_CRLB_values_2.append(CRLB_values_2)
            all_percent_CRLB_values_2.append(percent_CRLB_values_2)

        # Plot the CRLB values for the current V dataset
        axs[0, 0].plot(tr_values, CRLB_values_1, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize=10)
        axs[0, 1].plot(tr_values, CRLB_values_2, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize=10)
        axs[1, 0].plot(tr_values, percent_CRLB_values_1, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize=10)
        axs[1, 1].plot(tr_values, percent_CRLB_values_2, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize=10)

    # Calculate the average conc values for all volunteers
    if all_CRLB_values_1:
        avg_CRLB_values_1 = np.mean(all_CRLB_values_1, axis=0)
        axs[0, 0].plot(tr_values, avg_CRLB_values_1, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)
    if all_CRLB_values_2:
        avg_CRLB_values_2 = np.mean(all_CRLB_values_2, axis=0)
        axs[0, 1].plot(tr_values, avg_CRLB_values_2, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)
    if all_percent_CRLB_values_1:
        avg_percent_CRLB_values_1 = np.mean(all_percent_CRLB_values_1, axis=0)
        axs[1, 0].plot(tr_values, avg_percent_CRLB_values_1, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)
    if all_percent_CRLB_values_2:
        avg_percent_CRLB_values_2 = np.mean(all_percent_CRLB_values_2, axis=0)
        axs[1, 1].plot(tr_values, avg_percent_CRLB_values_2, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)

    global_fontsize = 16
    axs[0, 0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0, 0].set_ylabel('CRLB (mM)', fontsize=global_fontsize)
    axs[0, 0].set_title(f'Similar scan time, comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0, 0].grid(True)
    axs[0, 0].set_xticks([2, 5, 8])
    axs[0, 0].tick_params(axis='both', labelsize=global_fontsize)

    axs[0, 1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0, 1].set_ylabel('CRLB (mM)', fontsize=global_fontsize)
    axs[0, 1].set_title(f'Same number of acqs., comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks([2, 5, 8])
    axs[0, 1].tick_params(axis='both', labelsize=global_fontsize)

    axs[1, 0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1, 0].set_ylabel('Relative CRLB (%)', fontsize=global_fontsize)
    axs[1, 0].set_title(f'Similar scan time, comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks([2, 5, 8])
    axs[1, 0].tick_params(axis='both', labelsize=global_fontsize)

    axs[1, 1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1, 1].set_ylabel('Relative CRLB (%)', fontsize=global_fontsize)
    axs[1, 1].set_title(f'Same number of acqs., comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[1, 1].grid(True)
    axs[1, 1].set_xticks([2, 5, 8])
    axs[1, 1].tick_params(axis='both', labelsize=global_fontsize)

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_CRLBplot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0, 0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0, 0].legend(handles1s, labels1s, fontsize=13)

    # Display the plot in Streamlit
    st.pyplot(fig)


    ## CRLB test plots


    # Define volunteers, TRs, and Metabolites
    # metabs = datasets['V1']['TR2_064']['Metab']
    metabs = list(datasets['V1']['TR2_064'].iloc[:,0].unique())
    selected_metab_CRLB = st.selectbox('Select a metabolite to plot', metabs, key='CRLB_metabolite', index=metabs.index('NAA') if 'NAA' in metabs else 0)
    # Initialize dictionaries to store the combined data for each metabolite
    mM_data = {metab: [] for metab in metabs}
    mM_CRLB_data = {metab: [] for metab in metabs}

    # Iterate through each volunteer (V1, V2, V3, V4, V5)
    for vol in datasets.keys():
        # Iterate through each TR (TR2_064, TR2_128, TR5_064, TR8_032, TR8_064)
        for tr in datasets[vol].keys():
            # Get the current dataset
            df = datasets[vol][tr]
            # Iterate through each row (metabolite) in the dataset
            for index, row in df.iterrows():
                # Extract the metabolite name
                metab = row['Metab']
                # Append the 'mM' and 'mM CRLB' values to the respective lists in the dictionaries
                mM_data[metab].append(row['mM'])
                mM_CRLB_data[metab].append(row['mM CRLB'])

    # Convert the dictionaries to DataFrames for easier manipulation and display
    mM_df = pd.DataFrame(mM_data)
    mM_CRLB_df = pd.DataFrame(mM_CRLB_data)

    # Calculate the median value for each metabolite
    median_mM = mM_df.median()

    # Define your threshold:
    Percentage = 30 # Default value is 30%
    threshold = median_mM * Percentage / 100

    # Initialize a dictionary to store the count of values exceeding the threshold for each metabolite
    exceeding_threshold_count = {metab: 0 for metab in metabs}

    # Iterate through each metabolite
    for metab in metabs:
        # Get the threshold value for the current metabolite
        metab_threshold = threshold[metab]
        # Count the number of values in mM_CRLB_df that exceed the threshold for the current metabolite
        exceeding_threshold_count[metab] = (mM_CRLB_df[metab] > metab_threshold).sum()

    # Find metabolites with 20% or more data points exceeding the threshold
    metabolites_exceeding_20 = [metab for metab, count in exceeding_threshold_count.items() if count >= int(round(0.2 * len(mM_df)))]

    # Display the metabolites with more than 20% of data points exceeding the threshold using markdown
    st.markdown("#### Metabolites with more than 20% of data points exceeding the threshold:")
    st.markdown(", ".join([f"**{metab}**" for metab in metabolites_exceeding_20]))

    def plot_metabolite_data_single_plot(metab):
        # Get the data for the selected metabolite
        mM_values = mM_df[metab]
        mM_CRLB_values = mM_CRLB_df[metab]
        threshold_value = threshold[metab]
        median_value = median_mM[metab]

        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 4))

        # Generate some jitter for horizontal spread
        jitter = 0.1
        
        # Plot the mM values with jitter
        ax.scatter(np.random.normal(1, jitter, len(mM_values)), mM_values, label='mM values')
        # Plot the mM CRLB values with jitter
        ax.scatter(np.random.normal(2, jitter, len(mM_CRLB_values)), mM_CRLB_values, label='mM CRLB values')

        # Plot the median value line
        ax.axhline(y=median_value, color='r', linestyle='-', label='Median')
        # Plot the threshold as a shaded region
        ax.fill_between([0.5, 2.5], 0, threshold_value, color='g', alpha=0.3, label='Threshold Region')

        # Set the title and labels
        max_conc = np.max(mM_values)
        if max_conc < 1: 
            start_val = -0.03
        else:
            start_val = -0.03 * max_conc

        ax.set_title(f'Concentration and CRLB comparison for: {metab}')
        ax.set_ylabel('Concentration (mM)')
        ax.set_ylim(bottom=start_val)  # Start y-axis at -0.02
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Concentration', 'Absolute CRLB'])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Show the plot
        plt.tight_layout()
        st.pyplot(fig)

    # Use the selected_metabolite_plot variable from the previous code
    plot_metabolite_data_single_plot(selected_metab_CRLB)

############################### FWHM ################################
def page_FWHM():
    st.header("Full Width at Half Maximum (FWHM)")

    st.markdown("""#### Explanation:""")
    st.markdown("""Here, you can see how different TRs and the number of acquisitions affect the spectral width of different metabolites, characterized as the FWHM parameter during fitting.""")

    st.markdown("""- The figure on the left represents when scan time was kept as similar as possible, while the figure on the right keeps the number of acquisitions the same.""")

    st.markdown("""### Select the volunteer and metabolite of your choice:""")

    # Create a multiselect box for the volunteers for plot with a "Select All" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'Select All')
    selected_Vs_FWHMplot = st.multiselect('Select a volunteer:', v_options, ['Select All'], key='plot_FWHM_v')

    # If "Select All" is selected, select all the V datasets
    if 'Select All' in selected_Vs_FWHMplot:
        selected_Vs_FWHMplot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    all_metabs_FWHM = list(datasets['V1']['TR2_064'].iloc[:,0].unique())

    # List of metabolites to exclude
    exclude_metabolites = ['Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30', 'Cr', 'PCr']

    # Filter out the excluded metabolites
    metabolites_FWHM = [metabolite for metabolite in all_metabs_FWHM if metabolite not in exclude_metabolites]

    selected_metabolite_plot = st.selectbox('Select a metabolite:', metabolites_FWHM, key='plot_metabolite', index=metabolites_FWHM.index('NAA') if 'NAA' in metabolites_FWHM else 0)

    # Define the corresponding TR values
    tr_values = [2, 5, 8]

    # Define the TR datasets to plot
    tr_datasets_1 = ['TR2_128', 'TR5_064', 'TR8_032']
    tr_datasets_2 = ['TR2_064', 'TR5_064', 'TR8_064']

    # Define the colors for each volunteer
    colors = {
        'V1': '#EC1313', #Red (CMYK)
        'V2': '#EC7F11', #Tangerine - Orange
        'V3': '#0DBE1E', #Dark pastel green
        'V4': '#151CDD', #Chrysler blue
        'V5': '#7B065C'  #Byzantium - Purple
    }

    marker_shapes = {
        'V1': 'P',
        'V2': 'D',
        'V3': '^',
        'V4': 's',
        'V5': 'X'
    }

    label_map = {'Cr+PCr': 'tCr'}

    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(top=1.2)

    # Initialize lists to store the FWHM values for all volunteers
    all_FWHM_values_1 = []
    all_FWHM_values_2 = []

    # Loop over the selected V datasets
    for selected_V_FWHMplot in selected_Vs_FWHMplot:
        FWHM_values_1 = []
        FWHM_values_2 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_FWHMplot][tr_dataset_1]
            df_2 = datasets[selected_V_FWHMplot][tr_dataset_2]

            # Initialize the combined FWHM values
            combined_FWHM_value_1 = 0
            combined_FWHM_value_2 = 0

            # Filter the DataFrame based on the current metabolite
            filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == selected_metabolite_plot]

            # Get the FWHM value and add it to the combined FWHM value
            if not filtered_df_plot_1.empty:
                FWHM_value_1 = filtered_df_plot_1['FWHM'].values[0]
                combined_FWHM_value_1 += FWHM_value_1
            if not filtered_df_plot_2.empty:
                FWHM_value_2 = filtered_df_plot_2['FWHM'].values[0]
                combined_FWHM_value_2 += FWHM_value_2

            # Append the combined FWHM values to the lists
            if combined_FWHM_value_1:
                FWHM_values_1.append(combined_FWHM_value_1)
            if combined_FWHM_value_2:
                FWHM_values_2.append(combined_FWHM_value_2)

        # Add the FWHM values to the lists for all volunteers
        if FWHM_values_1:
            all_FWHM_values_1.append(FWHM_values_1)
        if FWHM_values_2:
            all_FWHM_values_2.append(FWHM_values_2)

        # Plot the FWHM values for the current V dataset
        axs[0].plot(tr_values, FWHM_values_1, marker=marker_shapes[selected_V_FWHMplot], label=f'Volunteer {selected_V_FWHMplot[1]}', color=colors[selected_V_FWHMplot], linewidth=4, markersize = 10)
        axs[1].plot(tr_values, FWHM_values_2, marker=marker_shapes[selected_V_FWHMplot], label=f'Volunteer {selected_V_FWHMplot[1]}', color=colors[selected_V_FWHMplot], linewidth=4, markersize = 10)

    # Calculate the average conc values for all volunteers
    if all_FWHM_values_1:
        avg_FWHM_values_1 = np.mean(all_FWHM_values_1, axis=0)
        axs[0].plot(tr_values, avg_FWHM_values_1, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)
    if all_FWHM_values_2:
        avg_conc_values_2 = np.mean(all_FWHM_values_2, axis=0)
        axs[1].plot(tr_values, avg_conc_values_2, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)

    global_fontsize = 16


    axs[0].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[0].set_ylabel('FWHM', fontsize=global_fontsize)
    axs[0].set_title(f'Similar scan time: FWHM comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0].grid(True)
    axs[0].set_xticks([2, 5, 8])
    axs[0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1].set_xlabel('TR (s)', fontsize=global_fontsize)
    axs[1].set_ylabel('FWHM', fontsize=global_fontsize)
    axs[1].set_title(f'Same number of acqs.: FWHM comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[1].grid(True)
    #axs[1].set_facecolor('black')
    axs[1].set_xticks([2, 5, 8])
    axs[1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_FWHMplot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0].legend(handles1s, labels1s, fontsize = 13)

    # Display the plot in Streamlit
    st.pyplot(fig)

############################ T1 fitting #############################
def page_T1_fit():
    st.header("T$_1$ Fitting")

    st.markdown("""#### Explanation:""")
    st.markdown("""This section accompanies Figure 5 of the manuscript.""")
    st.markdown("""Here, we dive into extracting T$_1$ fit values from concentration values for different metabolites.""")
    st.markdown("""- The first section allows you to display the fits for a specific metabolite and calculates the average and standard deviation T$_1$ values.""")
    st.markdown("""- Fits were done on individual volunteer data sets. The fit parameters were averaged across volunteers, and the standard deviation is used as the uncertainty.""")
    st.markdown("""- The fit equation used is the classic saturation recovery equation: M$_0$*(1 - exp(-TR/T$_1$))""")
    st.markdown("""- The second section displays the average T$_1$ value and associated standard deviation for multiple metabolites.""")


    st.markdown("""## Single metabolite fit""")
    st.markdown("""### Select the volunteer and metabolite of your choice:""")

    # Fit function
    def sat_rec(TR, M0, T1):
        return M0*(1-np.exp(-TR/T1))

    # TRs
    TRs = [2, 5, 8]

    # linespace for TRs
    TR_fit = np.linspace(0, 10, 250)  # Generate a smooth TR range for the fit

    # Initial guesses for M0 and T1
    params_initial_guess = [10, 1.2]


    # Get a list of all metabolites
    all_metabolites = datasets['V1']['TR2_064'].iloc[:, 0].tolist()

    # List of metabolites to exclude
    exclude_metabolites = ['Ala', 'Glc', 'Glyc', 'MM12', 'MM14', 'MM16', 'MM21', 'MM39', 'NAAG', 'bHB', 'MM09', 'MM30', 'Cr', 'PCr', 'GPC', 'PCh']

    # Filter out the excluded metabolites
    metabolites = [metabolite for metabolite in all_metabolites if metabolite not in exclude_metabolites]


    # Add the combined metabolite to the list
    if 'GPC+PCh' not in metabolites:
        metabolites.append('GPC+PCh')

    # Let the user select the metabolite
    selected_metabolite = st.selectbox('Select a metabolite:', metabolites, index=metabolites.index('NAA'))

    T1_array = []
    M0_array = []

    for v in ['V1', 'V2', 'V3', 'V4', 'V5']:
        # Check if the selected metabolite is the combined one
        if selected_metabolite == 'GPC+PCh':
            metabolites_to_combine = ['GPC', 'PCh']
        else:
            metabolites_to_combine = [selected_metabolite]

        combined_y_values = [0, 0, 0]
        for metabolite_to_combine in metabolites_to_combine:
            # Select the row corresponding to the current metabolite
            y_values = [datasets[v][tr]['mM'][datasets[v][tr].iloc[:, 0] == metabolite_to_combine].values[0] for tr in ['TR2_128', 'TR5_064', 'TR8_064']]
            combined_y_values = [sum(x) for x in zip(combined_y_values, y_values)]

        p_opt, p_cov = curve_fit(sat_rec, TRs, combined_y_values, p0=params_initial_guess)
        T1_array.append(p_opt[1])
        M0_array.append(p_opt[0])

    mean_T1 = np.mean(T1_array)
    std_T1 = np.std(T1_array)
    mean_M0 = np.mean(M0_array)

    st.markdown(f"""#### Mean T$_1$ = {mean_T1:.4f} s""")
    st.markdown(f"""#### Standard Deviation = {std_T1:.4f} s""")



    # Create a dictionary that maps the original labels to the new labels
    label_map = {'GPC+PCh': 'tCho', 'Cr+PCr': 'tCr'}

    fig = plt.figure(figsize=(10, 6))
    markers = ['v', 'o', 'd', '*', '>']
    colors = ['#0c2343', '#2e86ab', '#eeb868', '#c83349', '#5b7065']

    plt.plot(TR_fit, sat_rec(TR_fit, mean_M0, mean_T1), color='#0c2343', linewidth=10, alpha=1, label='Average Fit')
    for i, v in enumerate(['V1', 'V2', 'V3', 'V4', 'V5']):
        # Check if the selected metabolite is the combined one
        if selected_metabolite == 'GPC+PCh':
            metabolites_to_combine = ['GPC', 'PCh']
        else:
            metabolites_to_combine = [selected_metabolite]

        combined_y_values = [0, 0, 0]
        for metabolite_to_combine in metabolites_to_combine:
            # Select the row corresponding to the current metabolite
            y_values = [datasets[v][tr]['mM'][datasets[v][tr].iloc[:, 0] == metabolite_to_combine].values[0] for tr in ['TR2_128', 'TR5_064', 'TR8_064']]
            combined_y_values = [sum(x) for x in zip(combined_y_values, y_values)]

        p_opt, p_cov = curve_fit(sat_rec, TRs, combined_y_values, p0=params_initial_guess)
        plt.scatter(TRs, combined_y_values, s=100, color=colors[i], alpha=0.7, marker=markers[i])
        plt.plot(TR_fit, sat_rec(TR_fit, *p_opt), color=colors[i], linewidth=5, alpha=0.3, label=f'Fit for {v}')

    plt.xlabel('TR [s]', fontsize=20)
    plt.ylabel('Apparent Conc. [mM]', fontsize=20)
    plt.title(f'T$_1$ fits for {label_map.get(selected_metabolite, selected_metabolite)}', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    st.pyplot(fig)

    st.markdown("""## Many metabolite T$_1$ comparison""")
    st.markdown("""### Select your choice of metabolites:""")
    
    # Let the user select multiple metabolites
    selected_metabolites = st.multiselect('Select metabolites:', metabolites, default=['NAA','GPC+PCh','Cr+PCr','mI','Glu'])

    T1_values = []
    T1_errs = []

    for selected_metabolite in selected_metabolites:
        T1_array = []
        for v in ['V1', 'V2', 'V3', 'V4', 'V5']:
            # Check if the selected metabolite is the combined one
            if selected_metabolite == 'GPC+PCh':
                metabolites_to_combine = ['GPC', 'PCh']
            else:
                metabolites_to_combine = [selected_metabolite]

            combined_y_values = [0, 0, 0]
            for metabolite_to_combine in metabolites_to_combine:
                # Select the row corresponding to the current metabolite
                y_values = [datasets[v][tr]['mM'][datasets[v][tr].iloc[:, 0] == metabolite_to_combine].values[0] for tr in ['TR2_128', 'TR5_064', 'TR8_064']]
                combined_y_values = [sum(x) for x in zip(combined_y_values, y_values)]

            p_opt, p_cov = curve_fit(sat_rec, TRs, combined_y_values, p0=params_initial_guess)
            T1_array.append(p_opt[1])

        mean_T1 = np.mean(T1_array)
        std_T1 = np.std(T1_array)

        T1_values.append(mean_T1)
        T1_errs.append(std_T1)


    # Create a dictionary that maps the original labels to the new labels
    label_map = {'GPC+PCh': 'tCho', 'Cr+PCr': 'tCr'}

    # Create a figure and axis for the error bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the error bar plot
    ax.errorbar(range(1, len(selected_metabolites) + 1), T1_values, yerr=T1_errs, fmt='o', markersize=8, capsize=5, color='#0c2343')

    # Set labels for the x-axis
    ax.set_xticks(range(1, len(selected_metabolites) + 1))
    ax.set_xticklabels([label_map.get(metabolite, metabolite) for metabolite in selected_metabolites], fontsize=20)

    # Set labels for the y-axis
    ax.set_ylabel('T$_1$ relaxation times (s)', fontsize=20)
    ax.set_xlabel('Metabolites', fontsize=20)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=15)

    # Add horizontal grid lines
    ax.yaxis.grid(True)

    # Shade every second metabolite plotted
    for i in range(len(selected_metabolites)):
        if i % 2 == 0:  # Check if the index is even
            ax.axvspan(i + 0.5, i + 1.5, facecolor='gray', alpha=0.2)

    # Set the limits of the x-axis
    ax.set_xlim(0.5, len(selected_metabolites) + 0.5)

    st.pyplot(fig)



# Define the pages
pages = {
    "Introduction": page_intro,
    "Data Representation": vol_dat,
    "SNR": page_snr,
    "SNR per unit time": page_snr_per_time,
    "Concentration": page_concentration,
    "CRLB": page_CRLB,
    "FWHM": page_FWHM,
    "T1 Fitting": page_T1_fit,

}

# Create a select box in the sidebar for navigation
selected_page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Display the selected page
pages [selected_page]()
