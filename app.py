
import io
import os
import glob
import base64
import time
import threading
from datetime import datetime


import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

from src.pdf_gen import PDF
from src.sq_grid import square_grid

dataset_path_list = glob.glob(os.path.join('data', '*.csv'))

# Inerciales comentadas

variables = [
    #             'imuAx', 'imuAy', 'imuAz', 'imuCx',
    #            'imuCy', 'imuCz', 'imuGx', 'imuGy', 'imuGz',
    'T_1', 'T_2',  'T_3', 'P', 'A_1', 'A_2', 'Time_Data', 'RH',
    'NH3', 'CO', 'NO', 'C3H8', 'C4H10', 'CH4', 'H2', 'C2H5OH'
]

_variable_verb = [
    #    'Accelerometer x', 'Accelerometer y', 'Accelerometer z', 'Mag field x',
    #    'Mag field y', 'Mag field z', 'Gyroscope x', 'Gyroscope y', 'Gyroscope z',
    'Temperature 1', 'Temperature 2', 'Temperature 3', 'Preassure', 'Altitude 1', 'Altitude 2', 'Time', 'Rel humidity',
    'NH3', 'CO', 'NO', 'C3H8', 'C4H10', 'CH4', 'H2', 'C2H5OH'
]

_variable_unit = [
    #                  'g', 'g', 'g', ' μT',
    #                  ' μT', ' μT', '°', '°', '°',
    '°C',  '°C', '°C', 'Pa', 'm', 'm', 's', '%',
    'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm'
]

_map_styles = ["stamen-terrain", "open-street-map",
               "carto-positron", "carto-darkmatter"]

_colorscales = px.colors.named_colorscales()
_colorscales.remove('viridis')
_colorscales.insert(0, 'viridis')

v_df = pd.DataFrame(
    {'verbose': _variable_verb, 'unit': _variable_unit}, index=variables)

# -----------------------------MAIN------------------------------



st.title('Airborne measurement visualization tool cloud')
st.image(Image.open(os.path.join('assets', 'flag.webp')), width=40)

# --------------------------SIDEBAR--------------------------

st.sidebar.image(Image.open(os.path.join('assets', 'simple-logo-azul.png')),
                 use_column_width=True)

st.sidebar.title('Options')

st.sidebar.subheader('General')

dataset_path_selected = st.sidebar.multiselect(
    'Datasets', dataset_path_list, format_func=lambda text: datetime.strptime(text[6:12], r"%d%m%y").strftime(r'%d-%b-%Y'))
variable = st.sidebar.selectbox(
    'Variable', variables, format_func=lambda var: v_df.verbose[var])
basemap = st.sidebar.selectbox(
    'Basemap', _map_styles, 0
)
colorscale = st.sidebar.selectbox(
    'Colorscale', _colorscales
)

st.sidebar.subheader('Grid')
secondary_grid = st.sidebar.checkbox('square grid')
grid_points = st.sidebar.checkbox('show points in grid')

if secondary_grid:
    tile_side = st.sidebar.slider('Square side [deg]', 0.05, 0.2, 0.1, 0.01)
if (len(dataset_path_selected) > 0) and not secondary_grid:
    grid_size = st.sidebar.slider('grid density', 8, 18, 12)

_airplanes = [
    'airplane.jpeg'
]

selected_airplane = _airplanes[0]


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


st.sidebar.title('About')
st.sidebar.image(Image.open(os.path.join('assets', 'airplane.jpg')),
                 use_column_width=True,  caption='FAC-Caravan airborne measurenment platform')


# --------------------------SIDEBAR--------------------------


@st.cache
def load_dataset(filename):
    return pd.read_csv(filename)


if len(dataset_path_selected) > 0:
    loaded_data = load_dataset(dataset_path_selected[0])
    for dataset in dataset_path_selected[1:]:
        loaded_data = loaded_data.append(
            load_dataset(dataset), ignore_index=True)

if len(dataset_path_selected) == 0:
    st.subheader('Select one or more datasets')
# ----------------------------PUNTOS----------------------------

if len(dataset_path_selected) > 0:
    st.subheader('Points')

    figs_title = 'Dataset Selected plot'

@st.cache
def get_points_fig(dataframe, variable):
    fig = px.scatter_mapbox(dataframe, lat="lat", lon="lot", opacity=0.5,
                            color=variable, color_continuous_scale=colorscale, labels={
                                variable: f"{v_df.loc[variable].verbose} [{v_df.loc[variable].unit}]"
                            })
    fig.update_layout(mapbox_style=basemap,
                      margin={"r": 0, "t": 30, "l": 0, "b": 15})
    fig.update_layout(title_text=figs_title, title_x=0.45)
    return fig



@st.cache
def time_series_plot(dataframe, variable):
    _title = f'Time series with {v_df.loc[variable].verbose} (colored)'
    fig = go.Figure(data=go.Scatter(x=dataframe['Time_Data'], y=dataframe['A_1'], mode='markers',
                                    marker={
        'color': dataframe[f'{variable}'],
        'colorscale': colorscale
    }, hovertemplate="Time: %{x}<br>" + "Altitude: %{y} m <br>" + "<extra></extra>"))
    fig.update_layout(title_text=_title, title_x=0.45, xaxis_title="Time [s]",
                      yaxis_title="Altitude [m]",)
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 15})
    return fig


if len(dataset_path_selected) > 0:

    points_fig = get_points_fig(loaded_data, variable)
    height_plot = time_series_plot(loaded_data, variable)

    st.plotly_chart(points_fig)
    st.plotly_chart(height_plot)

# ---------------------------CONTORNO----------------------------
# st.subheader('Contorno')


# @st.cache
# def get_contour_fig(dataframe, variable):
#     fig = go.Figure(go.Densitymapbox(lat=dataframe.lat,
#                                      lon=dataframe.lot,
#                                      z=dataframe[f"{variable}"],
#                                      radius=15))
#     fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=-74)
#     fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#     return fig


# if len(dataset_path_selected) > 0:
#     contour_fig = get_contour_fig(loaded_data, variable)
#     st.plotly_chart(contour_fig)


# ----------------------------GRID----------------------------

if (len(dataset_path_selected) > 0) and not secondary_grid:
    st.subheader('Grid')


@st.cache
def get_grid_fig(dataframe, variable, grid_density):
    fig = ff.create_hexbin_mapbox(
        data_frame=loaded_data, lat="lat", lon="lot",
        nx_hexagon=grid_density, agg_func=np.mean, opacity=0.4,
        min_count=False, color=variable, color_continuous_scale=colorscale, show_original_data=grid_points,
        labels={
            "color": f"{v_df.loc[variable].verbose} [{v_df.loc[variable].unit}]"}
    )
    fig.update_layout(mapbox_style=basemap,
                      margin={"r": 0, "t": 30, "l": 0, "b": 0})
    fig.update_layout(title_text=figs_title, title_x=0.45)
    return fig


@st.cache
def get_grid2_fig(dataframe, variable):
    fig, _df = square_grid(loaded_data, opacity=0.4, color=variable, color_continuous_scale=colorscale,
                           labels={
                               "value": f"{v_df.loc[variable].verbose} [{v_df.loc[variable].unit}]"},
                           tile_side=tile_side, show_original_data=grid_points)
    fig.update_layout(mapbox_style=basemap,
                      margin={"r": 0, "t": 30, "l": 0, "b": 0})
    fig.update_layout(title_text=figs_title, title_x=0.45)
    return fig, _df


if len(dataset_path_selected) > 0:

    if secondary_grid:
        grid_fig, _df = get_grid2_fig(loaded_data, variable)
        st.plotly_chart(grid_fig)
    else:
        grid_fig = get_grid_fig(loaded_data, variable, grid_size)
        st.plotly_chart(grid_fig)


# --------------------------PDF-REPORT---------------------------


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download report</a>'


def get_table_download_link(df, download_name='file.csv', label='link'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{download_name}">{label}</a>'
    return href


if len(dataset_path_selected) > 0:
    st.title('Downloadable')

    generate_report_button = st.button('Generate report')

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.subheader('Report with current data')
        if generate_report_button:
            with st.spinner('Generanting download link'):

                timestamp = str(time.time())[:5]

                points_file = os.path.join('images', f'points-{timestamp}.png')
                grid_file = os.path.join('images', f'grid-{timestamp}.png')

                points_fig.write_image(points_file)  # Write points image
                grid_fig.write_image(grid_file)  # Write grid image

                pdf = PDF()  # El ancho de la página es 190
                pdf.alias_nb_pages()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(95, 10, 'Datasets: ', 0, 0, 'L')
                pdf.cell(
                    95, 10, f'Variable: {v_df.verbose[variable]}', 0, 1, 'R')

                pdf.set_font('Arial', '', 10)  # datasets
                for dataset in dataset_path_selected:
                    pdf.cell(95, 7, f'{dataset[5:]}', 0, 1, 'L')

                # Imagen de puntos  # wrapper(w, h=0, txt='', border=0, ln=0, align='', fill=0, link='')
                # pdf.image(points_file, w=190)

                pdf.image(points_file, w=190)

                pdf.set_font('Arial', '', 12)
                pdf.cell(94, 8, 'Datos generales', 1, 0, 'C')
                pdf.cell(2)
                pdf.cell(94, 8, f'Datos {v_df.verbose[variable]}', 1, 1, 'C')

                pdf.cell(47, 8, 'Número de datos', 1, 0, 'C')
                pdf.cell(47, 8, f'{loaded_data.shape[0]}', 1, 0, 'C')
                pdf.cell(2)
                pdf.cell(47, 8, 'Mínimo', 1, 0, 'C')
                pdf.cell(
                    47, 8, f'{loaded_data[variable].min():.3f} {v_df.unit[variable]}', 1, 1, 'C')

                pdf.cell(47, 8, 'Rango de longitudes', 1, 0, 'C')
                pdf.cell(
                    47, 8, f' {loaded_data.lot.min():.5f} ; {loaded_data.lot.max():.5f}', 1, 0, 'C')
                pdf.cell(2)
                pdf.cell(47, 8, 'Máximo', 1, 0, 'C')
                pdf.cell(
                    47, 8, f'{loaded_data[variable].max():.3f} {v_df.unit[variable]}', 1, 1, 'C')

                pdf.cell(47, 8, 'Rango de latitudes', 1, 0, 'C')
                pdf.cell(
                    47, 8, f' {loaded_data.lat.min():.5f} ; {loaded_data.lat.max():.5f}', 1, 0, 'C')
                pdf.cell(2)
                pdf.cell(47, 8, 'Promedio', 1, 0, 'C')
                pdf.cell(
                    47, 8, f'{loaded_data[variable].mean():.3f} {v_df.unit[variable]}', 1, 1, 'C')

                pdf.cell(47, 8, '', 0, 0, 'C')
                pdf.cell(47, 8, '', 0, 0, 'C')
                pdf.cell(2)
                pdf.cell(47, 8, 'Desviación estándar', 1, 0, 'C')
                pdf.cell(
                    47, 8, f'{loaded_data[variable].std():.4f} {v_df.unit[variable]}', 1, 1, 'C')

                pdf.image(grid_file, w=190)

                os.remove(points_file)
                os.remove(grid_file)

                html = create_download_link(pdf.output(dest="S").encode(
                    "latin-1"), f"reporte-{variable}")  # Output PDF and create download link
                st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.subheader('Dataset csv')
        st.markdown(get_table_download_link(
            loaded_data, 'dataset.csv', 'download (csv)'), unsafe_allow_html=True)

    with col3:

        if secondary_grid:
            st.subheader('Grid data product for assimilation with LOTOS-EUROS')
            st.markdown(get_table_download_link(
                _df, 'grid.csv', 'download (csv)'), unsafe_allow_html=True)
