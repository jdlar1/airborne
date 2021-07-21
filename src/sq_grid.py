import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px


def find_limits(dataframe: pd.DataFrame, side: float = 0.01, time_div=15) -> Tuple[np.ndarray, np.ndarray]:
    """Finds lats and lots bounds arrays given a square side for the grid definition

    Args:
        dataframe (pd.DataFrame): Dataframe containing columns lat and lon.
        side (float, optional): Square grid side in decimal degrees. Defaults to 0.01.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with lats_bnds and lots_bnds
    """

    lots, lats = tuple(np.split(dataframe[['lot', 'lat']].values.T, 2, 0))

    x_min_max = np.array([lats.min(), lats.max()])
    y_min_max = np.array([lots.min(), lots.max()])

    n_sq_x = np.ceil((x_min_max[1] - x_min_max[0]) / side).astype(int)
    n_sq_y = np.ceil((y_min_max[1] - y_min_max[0]) / side).astype(int)

    x_upper_diff = n_sq_x * side - x_min_max[1] + x_min_max[0]
    y_upper_diff = n_sq_y * side - y_min_max[1] + y_min_max[0]

    c_x = x_min_max[0] - x_upper_diff / 2
    c_y = y_min_max[0] - y_upper_diff / 2

    lat_bnds = np.linspace(c_x, c_x + n_sq_x * side, n_sq_x + 1)
    lot_bnds = np.linspace(c_y, c_y + n_sq_y * side, n_sq_y + 1)

    return lat_bnds, lot_bnds


def _get_number_sides(dataframe: pd.DataFrame, tile_side: float = 0.1):

    df_x_ends = [dataframe.lot.min(), dataframe.lot.max()]
    df_y_ends = [dataframe.lat.min(), dataframe.lat.max()]

    deg_diff = np.array([
        df_x_ends[1] - df_x_ends[0],
        df_y_ends[1] - df_y_ends[0]
    ])

    tiles_num = (deg_diff // tile_side) + 1  # Number of squares in between
    grid_sides = tiles_num * tile_side

    _originx = df_x_ends[0] - ((grid_sides[0]-deg_diff[0]) / 2)
    _originy = df_y_ends[1] + ((grid_sides[1]-deg_diff[1]) / 2)
    origin = np.array([_originx, _originy])  # Left top corner of the grid

    _grid_limit_x = origin[0] + tiles_num[0] * tile_side
    _grid_limit_y = origin[1] - tiles_num[1] * tile_side
    # Limits counted from origin
    grid_limits = np.array([_grid_limit_x, _grid_limit_y])

    x_coordinates = np.linspace(origin[0], grid_limits[0], int(tiles_num[0]))
    y_coordinates = np.linspace(origin[1], grid_limits[1], int(tiles_num[1]))

    return x_coordinates, y_coordinates


def _geojson_gen(dataframe: pd.DataFrame = None, column: str = None, x_coords: list = None, y_coords: list = None):

    len_x = len(x_coords[:-1])
    len_y = len(y_coords[:-1])

    _geolist = []

    _grid_id = []
    _values = []

    geo_df = pd.DataFrame(columns=['grid_id', 'value'])

    for idx_x in range(len_x):
        for idx_y in range(len_y):

            name = f'{idx_x}-{idx_y}'

            tl = [x_coords[idx_x], y_coords[idx_y]]
            bl = [x_coords[idx_x], y_coords[idx_y+1]]
            br = [x_coords[idx_x+1], y_coords[idx_y + 1]]
            tr = [x_coords[idx_x+1], y_coords[idx_y]]

            _mask1 = (dataframe['lot'] >= x_coords[idx_x]) & (
                dataframe['lot'] < x_coords[idx_x+1])
            _mask2 = (dataframe['lat'] <= y_coords[idx_y]) & (
                dataframe['lat'] > y_coords[idx_y+1])

            mask = _mask1 & _mask2

            _catched = dataframe[mask]

            if len(_catched.index) > 0:
                geo_df = geo_df.append({
                    'grid_id': name,
                    'value': _catched[column].mean()
                }, ignore_index=True)
            else:
                geo_df = geo_df.append({'grid_id': name, 'value': np.nan},
                                       ignore_index=True)

            _geojson = {'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [[tl, bl, br, tr, tl]]
                        },
                        'id': name
                        }
            _geolist.append(_geojson)

    geojson = {
        'type': 'FeatureCollection',
        'features': _geolist,
    }

    return geo_df, geojson, [x_coords, y_coords]


def _zoom_center(lons: list = None, lats: list = None, lonlats: tuple = None,
                 format: str = 'lonlat', projection: str = 'mercator',
                 width_to_height: float = 2.0) -> (float, dict):
    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    print(zoom_center((-109.031387, -103.385460),
        (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    if projection == 'mercator':
        margin = 1.5
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )

    return zoom, center


def square_grid(
    dataframe: pd.DataFrame,
    title: str = None,
    labels: dict = None,
    color: str = None,
    tile_side: float = 0.05,
    color_continuous_scale: str = 'Viridis',
    opacity: float = 0.4,
    show_original_data: bool = False
):

    _x_vals, _y_vals = _get_number_sides(dataframe, tile_side=tile_side)

    df, geojson, _coords = _geojson_gen(
        dataframe, x_coords=_x_vals, y_coords=_y_vals, column=color
    )

    zoom, center = _zoom_center(
        lons=dataframe.lot,
        lats=dataframe.lat
    )

    _fig = px.choropleth_mapbox(
        df, geojson=geojson, color="value",
        locations="grid_id", featureidkey="id",
        mapbox_style="carto-positron", zoom=zoom, center=center, opacity=opacity,
        title=title, labels=labels, color_continuous_scale=color_continuous_scale)

    _fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 10})

    if show_original_data:
        original_fig = px.scatter_mapbox(dataframe, lat="lat", lon="lot", opacity=0.5,
                                         color=dataframe[color], color_continuous_scale="Viridis")

        original_fig.data[0].hoverinfo = "skip"
        original_fig.data[0].hovertemplate = None

        _fig.add_trace(original_fig.data[0])

    return _fig, df


if __name__ == "__main__":

    df = pd.read_csv(os.path.join('data', 'D131120.csv'))

    fig = square_grid(df, color='NH3')
    fig.show()
