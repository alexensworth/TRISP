import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from matplotlib import MatplotlibDeprecationWarning
from scipy.optimize import curve_fit

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

# Modify the SNR values for 'Cr+PCr'
#for volunteer in datasets:
#    for TR in datasets[volunteer]:
#        df = datasets[volunteer][TR]
#        if 'Cr+PCr' in df.iloc[:, 0].tolist():
#            cr_pcr_index = df.index[df.iloc[:, 0] == 'Cr+PCr'].tolist()[0]
#            cr_index = df.index[df.iloc[:, 0] == 'Cr'].tolist()[0]
#            pcr_index = df.index[df.iloc[:, 0] == 'PCr'].tolist()[0]
#            df.at[cr_pcr_index, "SNR"] = df.at[cr_index, "SNR"] + df.at[pcr_index, "SNR"]

############################ Introduction ###########################
def page_intro():
    st.markdown("""
                # An investigation of repetition time on MR spectroscopy """)
    st.markdown("""## Purpose:""")
    st.markdown("""### This study aims to determine the optimal balance ebtween scan time and repetition time that minimizes T1 weighting effects.""")
    st.markdown("""## Parameters:""")
    st.markdown(""" - 5 healthy controls (mean age 25 $\pm$ 2 years)""")
    st.markdown(""" - 3 T Philips Ingenia Elition X""")
    st.markdown(""" - semi-LASER localization""")
    st.markdown(""" - 30 x 20 x 13 mm$^3$ voxel in the posterior cingulate cortex (PCC)""")
    st.markdown("""
                This document reflects the data taken in this study. 
                Each page reflects various aspects of analysis conducted in this study.  """)
    st.markdown("""## How to use this tool:""")
    st.markdown(""" - Navigate to various analysis pages via the menu on the left""")
    st.markdown(""" - Upon selecting a page, the default selection of data is what is presented in the manuscript""")
    st.markdown(""" - Modify the selections by choosing different items in the drop down menus.""")
    

def vol_dat():
    st.header("Data representation")
    st.markdown("""#### Select the volunteer and TR of your choice to view the corresponding spectral fit and data table
                """)
    
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

    st.markdown(f"""### Key data for volunteer {selected_V_spec}, {selected_tr_label}""")
    
    # Display the selected dataset as a table
    selected_dataset = datasets[selected_V_spec][selected_tr]
    Cols_to_display = ['Metab','SNR','mM','mM CRLB', 'FWHM'];
    selected_dataset=selected_dataset[Cols_to_display]
    #rows_to_drop = []
    rows_to_drop = [11, 12, 13, 14, 15, 16, 17]
    selected_dataset = selected_dataset.drop(rows_to_drop)
    selected_dataset=selected_dataset.T
    # Format numeric values to 4 decimal places
    selected_dataset = selected_dataset.applymap(lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x)

    st.dataframe(selected_dataset)


    
############################ SNR content ###########################
def page_snr():
    st.header("SNR")

    st.markdown("""#### How the SNR of a specific metabolite changes across different repetition times and number of acquisitions""")
                
    # Create a select box for the datasets
    selected_V_snr = st.selectbox('Select a V dataset for SNR display', list(datasets.keys()), key='snr_v')

    # Create a select box for the TR datasets
    selected_TR_snr_label = st.selectbox('Select a TR dataset for SNR display', trs, key='snr_tr')
    selected_TR_snr = list(tr_mapping.keys())[list(tr_mapping.values()).index(selected_TR_snr_label)]

    # Assuming the first column contains the metabolite names for all sets of data
    metabolites = [metabolite for metabolite in datasets[selected_V_snr][selected_TR_snr].iloc[:,0].unique() if metabolite != 'Cr+PCr']
    
    # Create a select box for the metabolites for SNR display with 'NAA' as the default value
    selected_metabolite_snr = st.selectbox('Select a metabolite for SNR display', metabolites, key='snr_metabolite', index=metabolites.index('NAA') if 'NAA' in metabolites else 0)

    # Filter the DataFrame based on the selected metabolite
    filtered_df_snr = datasets[selected_V_snr][selected_TR_snr][datasets[selected_V_snr][selected_TR_snr].iloc[:, 0] == selected_metabolite_snr]

    # Display the SNR value for the selected metabolite
    snr_value = format(filtered_df_snr['SNR'].values[0], ".2f")
    st.text(f'The SNR value for {selected_metabolite_snr} in dataset {selected_V_snr} {selected_TR_snr} is {snr_value}')

    # Create a multiselect box for the volunteers for plot with a "Select All" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'Select All')
    selected_Vs_plot = st.multiselect('Select V datasets for plot', v_options, ['Select All'], key='plot_v')

    # If "Select All" is selected, select all the V datasets
    if 'Select All' in selected_Vs_plot:
        selected_Vs_plot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    selected_metabolite_plot = st.selectbox('Select a metabolite for plot', metabolites, key='plot_metabolite', index=metabolites.index('NAA') if 'NAA' in metabolites else 0)

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
    
    st.markdown("""##### The first row of figures represents the absolute SNR for this metabolite, while the second row has normalized the SNR to a TR of 8s.""")
    st.markdown("""##### The first column of figures represents when acquisition time was kept as similar as possible, while the second column keeps the number of acquisitions the same.""")
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

    axs[0, 0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[0, 0].set_ylabel('SNR', fontsize=global_fontsize)
    axs[0, 0].set_title(f'Similar acquisition time, SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[0, 0].grid(True)
    axs[0, 0].set_xticks([2, 5, 8])
    axs[0, 0].tick_params(axis='both', labelsize=global_fontsize);

    axs[0, 1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[0, 1].set_ylabel('SNR', fontsize=global_fontsize)
    axs[0, 1].set_title(f'Same number of acquisitions, SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks([2, 5, 8])
    axs[0, 1].tick_params(axis='both', labelsize=global_fontsize);

    axs[1, 0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[1, 0].set_ylabel('Normalized SNR', fontsize=global_fontsize)
    axs[1, 0].set_title(f'Similar acquisition time, normalized SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks([2, 5, 8])
    axs[1, 0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1, 1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[1, 1].set_ylabel('Normalized SNR', fontsize=global_fontsize)
    axs[1, 1].set_title(f'Similar number of acquisitions, normalized SNR comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize)
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

############################ SNR per unit time ###########################
    
def page_snr_per_time():
    st.header("SNR per unit scan time")

    st.markdown("""##### The SNR for a metabolite of your choice is divided by the scan time for each volunteer at the respective TR. The average is represented by a solid black line""")
    st.markdown("""##### The anticipated SNR is shown as a red dashed line, and it is based on a calculation of the number of acquisitions one gets at a different TR in the same time.""")

    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    # Define your times
    TR2_time = 264/60
    TR5_time = 340/60
    TR8_time = 544/60

    # Get a list of all metabolites
    #metabolites = datasets['V1']['TR2_064'].iloc[:, 0].tolist()
    metabolites = [metabolite for metabolite in datasets['V1']['TR2_064'].iloc[:,0].unique() if metabolite != 'Cr+PCr']

    # Let the user select the metabolite
    selected_metabolite = st.selectbox('Select a metabolite', metabolites, index=metabolites.index('NAA'))

    # Allow the user to select a "reference" TR
    tr_pt_mapping = {
    'TR2_128': 'TR=2s, 128acqs',
    'TR5_064': 'TR=5s, 64acqs',
    'TR8_064': 'TR=8s, 64acqs',
    }

    trs_pt = list(tr_pt_mapping.values())

    reference_TR_label = st.selectbox('Select a reference TR', trs_pt, index=2)
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
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Individuals',
                            markerfacecolor='#0c2343', markersize=25, alpha=0.7),
                    Line2D([0], [0], color='k', lw=6, label='Average'),
                    Line2D([0], [0], color='r', lw=6, linestyle='--', label='Calculated')]
    ax.legend(handles=legend_elements, fontsize=20)
    
    # Set x-axis labels to be 2, 5, and 8
    ax.set_xticks(range(len(line_values)))
    ax.set_xticklabels(['2', '5', '8'])

    plt.xlabel("Repetition time, TR [s]", fontsize=20)
    plt.ylabel("SNR per unit scan time [SNR/minute]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Display the plot in Streamlit
    st.pyplot(fig)

############################ Concentration content ############################

def page_concentration():
    st.header("Concentration")

    st.markdown("""#### How the reported concentration of a specific metabolite changes across different repetition times and number of acquisitions""")

    # Create a select box for the datasets
    selected_V_conc = st.selectbox('Select a V dataset for concentration display', list(datasets.keys()), key='conc_v')

    # Create a select box for the TR datasets
    selected_TR_conc_label = st.selectbox('Select a TR dataset for concentration display', trs, key='conc_tr')
    selected_TR_conc = list(tr_mapping.keys())[list(tr_mapping.values()).index(selected_TR_conc_label)]

    # Assuming the first column contains the metabolite names for all sets of data
    metabolites = datasets[selected_V_conc][selected_TR_conc].iloc[:,0].unique()

    # Create a select box for the metabolites for SNR display with 'NAA' as the default value
    selected_metabolite_conc = st.selectbox('Select a metabolite for concentration display', metabolites, key='conc_metabolite', index=metabolites.tolist().index('NAA') if 'NAA' in metabolites else 0)

    # Filter the DataFrame based on the selected metabolite
    filtered_df_conc = datasets[selected_V_conc][selected_TR_conc][datasets[selected_V_conc][selected_TR_conc].iloc[:, 0] == selected_metabolite_conc]

    # Display the SNR value for the selected metabolite
    conc_value = format(filtered_df_conc['mM'].values[0], ".2f")
    st.text(f'The concentration value for {selected_metabolite_conc} in dataset {selected_V_conc} {selected_TR_conc} is {conc_value}')

###### Plot the concentration of a metabolite for the volunteers and display averages #########


    # Create a multiselect box for the volunteers for plot with a "Select All" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'Select All')
    selected_Vs_plot = st.multiselect('Select V datasets for plot', v_options, ['Select All'], key='plot_v')  # Set 'Select All' as the default option

    # If "Select All" is selected, select all the V datasets
    if 'Select All' in selected_Vs_plot:
        selected_Vs_plot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    metabolites_conc_avg = list(datasets[selected_V_conc][selected_TR_conc].iloc[:,0].unique())

    # Add the combined metabolite to the list
    metabolites_conc_avg.append('GPC+PCh')

    selected_metabolite_plot = st.selectbox('Select a metabolite for plot', metabolites_conc_avg, key='plot_metabolite', index=metabolites_conc_avg.index('NAA') if 'NAA' in metabolites_conc_avg else 0)

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

    # Create a color map
    #cmap = plt.get_cmap('jet')

    # # Create a dictionary to map each metabolite to a color
    # color_dict = {}
    # for i, metabolite in enumerate(metabolites_conc_avg):
    #     color_dict[metabolite] = cmap(i / len(metabolites_conc_avg))


    st.markdown("""##### The apparent concentrations for the selected metabolite are shown for each selected volunteer where the graph on the left represents when acquisition time was kept as similar as possible, while the second column keeps the number of acquisitions the same.""")
    
    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(top=1.2)

    # Initialize lists to store the SNR values for all volunteers
    all_conc_values_1 = []
    all_conc_values_2 = []

    # Loop over the selected V datasets
    for selected_V_plot in selected_Vs_plot:
        conc_values_1 = []
        conc_values_2 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_plot][tr_dataset_1]
            df_2 = datasets[selected_V_plot][tr_dataset_2]

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
        axs[0].plot(tr_values, conc_values_1, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)
        axs[1].plot(tr_values, conc_values_2, marker=marker_shapes[selected_V_plot], label=f'Volunteer {selected_V_plot[1]}', color=colors[selected_V_plot], linewidth=4, markersize = 10)

    # Calculate the average conc values for all volunteers
    if all_conc_values_1:
        avg_conc_values_1 = np.mean(all_conc_values_1, axis=0)
        axs[0].plot(tr_values, avg_conc_values_1, marker='.', linestyle='--', label=f'Average {selected_metabolite_plot}', color='black', linewidth=4)
    if all_conc_values_2:
        avg_conc_values_2 = np.mean(all_conc_values_2, axis=0)
        axs[1].plot(tr_values, avg_conc_values_2, marker='.', linestyle='--', label=f'Average {selected_metabolite_plot}', color='black', linewidth=4)

    global_fontsize = 16

    axs[0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[0].set_ylabel('Apparent concentration', fontsize=global_fontsize)
    axs[0].set_title(f'Similar acq. time: reported conc. comparison for {selected_metabolite_plot}', fontsize=global_fontsize+4)
    axs[0].grid(True)
    axs[0].set_xticks([2, 5, 8])
    axs[0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[1].set_ylabel('Apparent concentration', fontsize=global_fontsize)
    axs[1].set_title(f'Same number of acqs.: reported conc. comparison for {selected_metabolite_plot}', fontsize=global_fontsize+4)
    axs[1].grid(True)
    #axs[1].set_facecolor('black')
    axs[1].set_xticks([2, 5, 8])
    axs[1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_plot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0].legend(handles1s, labels1s, fontsize = 13)

    # Display the plot in Streamlit
    st.pyplot(fig)

    ###### Plot the average of the metabolite for many metabolites #########


    # Create the subplots
    fig2, axs2 = plt.subplots(1, 2, figsize=(20, 6))
    plt.subplots_adjust(top=1.2)

    # Assuming the first column contains the metabolite names for all sets of data
    metabolites_conc_avg = list(datasets[selected_V_conc][selected_TR_conc].iloc[:,0].unique())

    # Add the combined metabolite to the list
    metabolites_conc_avg.append('GPC+PCh')

    selected_metabolites_plot = st.multiselect('Select metabolites for plot', metabolites_conc_avg, ['NAA', 'Cr+PCr', 'GPC+PCh', 'mI', 'Glu'], key='average_plot_metabolite')

    st.markdown("""##### Similar to the above graphs, however, the concentrations for a selection of metabolites are averaged, and normalized to a TR of 8s.""")
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
        for selected_V_plot in all_Vs:
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
                    df_1 = datasets[selected_V_plot][tr_dataset_1]
                    df_2 = datasets[selected_V_plot][tr_dataset_2]

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
            axs2[0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
            axs2[0].set_ylabel('Normalized apparent concentration', fontsize=global_fontsize)
            axs2[0].set_title(f'Similar acq time, normalized reported concs', fontsize=global_fontsize+4)
            axs2[0].grid(True)
            axs2[0].set_xticks([2, 5, 8])
            axs2[0].tick_params(axis='both', labelsize=global_fontsize);
        if all_conc_values_2:
            avg_conc_values_2 = np.mean(all_conc_values_2, axis=0)
            # Normalize the average values by the average value at TR8
            avg_conc_values_2 = avg_conc_values_2 / avg_conc_values_2[-1]
            axs2[1].plot(tr_values, avg_conc_values_2, marker='o', linestyle='-', label=f'{selected_metabolite_plot}', color=color_dict[selected_metabolite_plot], linewidth=5, markersize=12)
            axs2[1].set_facecolor('black')
            axs2[1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
            axs2[1].set_ylabel('Normalized apparent concentration', fontsize=global_fontsize)
            axs2[1].set_title(f'Same number of acqs, normalized reported concs', fontsize=global_fontsize+4)
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




    # Create a color map
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 
    #'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 
    #'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 
    #'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 
    #'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 
    #'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 
    #'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 
    #'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 
    #'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
    #'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 
    #'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 
    #'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 
    #'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 
    #'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

############################ CRLB content ###########################
def page_CRLB():
    st.header("CRLB")

        # Create a multiselect box for the volunteers for plot with a "Select All" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'Select All')
    selected_Vs_CRLBplot = st.multiselect('Select V datasets for plot', v_options, ['Select All'], key='plot_CRLB_v')

    # If "Select All" is selected, select all the V datasets
    if 'Select All' in selected_Vs_CRLBplot:
        selected_Vs_CRLBplot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    metabolites_CRLB = list(datasets['V1']['TR2_064'].iloc[:,0].unique())

    # Add the combined metabolite to the list
    #metabolites_CRLB.append('GPC+PCh')

    selected_metabolite_plot = st.selectbox('Select a metabolite for plot', metabolites_CRLB, key='plot_metabolite', index=metabolites_CRLB.index('NAA') if 'NAA' in metabolites_CRLB else 0)

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

    # Initialize lists to store the CRLB values for all volunteers
    all_CRLB_values_1 = []
    all_CRLB_values_2 = []

    # Loop over the selected V datasets
    for selected_V_CRLBplot in selected_Vs_CRLBplot:
        CRLB_values_1 = []
        CRLB_values_2 = []
        # Loop over the TR datasets
        for tr_dataset_1, tr_dataset_2 in zip(tr_datasets_1, tr_datasets_2):
            # Get the selected DataFrame
            df_1 = datasets[selected_V_CRLBplot][tr_dataset_1]
            df_2 = datasets[selected_V_CRLBplot][tr_dataset_2]

            # Initialize the combined CRLB values
            combined_CRLB_value_1 = 0
            combined_CRLB_value_2 = 0

            # Filter the DataFrame based on the current metabolite
            filtered_df_plot_1 = df_1[df_1.iloc[:, 0] == selected_metabolite_plot]
            filtered_df_plot_2 = df_2[df_2.iloc[:, 0] == selected_metabolite_plot]

            # Get the CRLB value and add it to the combined CRLB value
            if not filtered_df_plot_1.empty:
                CRLB_value_1 = filtered_df_plot_1['mM CRLB'].values[0]
                combined_CRLB_value_1 += CRLB_value_1
            if not filtered_df_plot_2.empty:
                CRLB_value_2 = filtered_df_plot_2['mM CRLB'].values[0]
                combined_CRLB_value_2 += CRLB_value_2

            # Append the combined CRLB values to the lists
            if combined_CRLB_value_1:
                CRLB_values_1.append(combined_CRLB_value_1)
            if combined_CRLB_value_2:
                CRLB_values_2.append(combined_CRLB_value_2)

        # Add the CRLB values to the lists for all volunteers
        if CRLB_values_1:
            all_CRLB_values_1.append(CRLB_values_1)
        if CRLB_values_2:
            all_CRLB_values_2.append(CRLB_values_2)

        # Plot the CRLB values for the current V dataset
        axs[0].plot(tr_values, CRLB_values_1, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize = 10)
        axs[1].plot(tr_values, CRLB_values_2, marker=marker_shapes[selected_V_CRLBplot], label=f'Volunteer {selected_V_CRLBplot[1]}', color=colors[selected_V_CRLBplot], linewidth=4, markersize = 10)

    # Calculate the average conc values for all volunteers
    if all_CRLB_values_1:
        avg_CRLB_values_1 = np.mean(all_CRLB_values_1, axis=0)
        axs[0].plot(tr_values, avg_CRLB_values_1, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)
    if all_CRLB_values_2:
        avg_conc_values_2 = np.mean(all_CRLB_values_2, axis=0)
        axs[1].plot(tr_values, avg_conc_values_2, marker='.', linestyle='--', label=f'Average {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', color='black', linewidth=4)

    global_fontsize = 16


    axs[0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[0].set_ylabel('CRLB', fontsize=global_fontsize)
    axs[0].set_title(f'Similar acq. time: CRLB comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0].grid(True)
    axs[0].set_xticks([2, 5, 8])
    axs[0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[1].set_ylabel('CRLB', fontsize=global_fontsize)
    axs[1].set_title(f'Same number of acqs.: CRLB comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[1].grid(True)
    #axs[1].set_facecolor('black')
    axs[1].set_xticks([2, 5, 8])
    axs[1].tick_params(axis='both', labelsize=global_fontsize);

    # Add the legend only if at least one V dataset is selected
    if selected_Vs_CRLBplot:
        # Get the handles and labels for the first subplot
        handles1, labels1 = axs[0].get_legend_handles_labels()

        # Sort them by labels
        labels1s, handles1s = zip(*sorted(zip(labels1, handles1), key=lambda t: t[0]))

        # Set the legend
        axs[0].legend(handles1s, labels1s, fontsize = 13)

    # Display the plot in Streamlit
    st.pyplot(fig)

    
def page_FWHM():
    st.header("FWHM")

        # Create a multiselect box for the volunteers for plot with a "Select All" option
    v_options = list(datasets.keys())
    v_options.insert(0, 'Select All')
    selected_Vs_FWHMplot = st.multiselect('Select V datasets for plot', v_options, ['Select All'], key='plot_FWHM_v')

    # If "Select All" is selected, select all the V datasets
    if 'Select All' in selected_Vs_FWHMplot:
        selected_Vs_FWHMplot = list(datasets.keys())

    # Create a select box for the metabolites for plot with 'NAA' as the default value
    metabolites_FWHM = list(datasets['V1']['TR2_064'].iloc[:,0].unique())

    selected_metabolite_plot = st.selectbox('Select a metabolite for plot', metabolites_FWHM, key='plot_metabolite', index=metabolites_FWHM.index('NAA') if 'NAA' in metabolites_FWHM else 0)

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


    axs[0].set_xlabel('TR (seconds)', fontsize=global_fontsize)
    axs[0].set_ylabel('FWHM', fontsize=global_fontsize)
    axs[0].set_title(f'Similar acq. time: FWHM comparison for {label_map.get(selected_metabolite_plot, selected_metabolite_plot)}', fontsize=global_fontsize+4)
    axs[0].grid(True)
    axs[0].set_xticks([2, 5, 8])
    axs[0].tick_params(axis='both', labelsize=global_fontsize);

    axs[1].set_xlabel('TR (seconds)', fontsize=global_fontsize)
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

############################ T1 fit content ###########################
def page_T1_fit():
    st.header("T1 Fitting")
    st.markdown("""#### Using our TR data to determine the T1 relaxation times for metabolites via saturation recovery.""")

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
    metabolites = datasets['V1']['TR2_064'].iloc[:, 0].tolist()

    # Add the combined metabolite to the list
    if 'GPC+PCh' not in metabolites:
        metabolites.append('GPC+PCh')

    # Let the user select the metabolite
    selected_metabolite = st.selectbox('Select a metabolite', metabolites, index=metabolites.index('NAA'))

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

    st.write(f"Mean T<sub>1</sub>: {mean_T1:.4f} s", unsafe_allow_html=True)
    st.write(f"Std Dev T<sub>1</sub>: {std_T1: .4f} s", unsafe_allow_html=True)

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
    plt.legend()
    
    st.pyplot(fig)


    # Let the user select multiple metabolites
    selected_metabolites = st.multiselect('Select metabolites', metabolites, default=['NAA','GPC+PCh','Cr+PCr','mI','Glu'])

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
    ax.set_ylabel('T$_1$ relaxation times [s]', fontsize=20)
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
    "Indiv. volunteer Data": vol_dat,
    "SNR": page_snr,
    "SNR per unit time": page_snr_per_time,
    "Concentration": page_concentration,
    "CRLB": page_CRLB,
    "FWHM": page_FWHM,
    "T1 Fits": page_T1_fit,

}

# Create a select box in the sidebar for navigation
selected_page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Display the selected page
pages [selected_page]()

