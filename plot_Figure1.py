# coding = 'utf-8'
import numpy as np
import geopandas as gpd
import xarray as xr
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb, to_hex
mpl.rcParams['font.family'] = 'Arial'

# load administrative shapefiles
county_shp = gpd.read_file('tl_2020_us_county.shp')   # replace with your own path
county_shp = county_shp.to_crs('EPSG:4326')
county_shp = county_shp[~county_shp['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]

state_shp = gpd.read_file('tl_2020_us_state.shp')   # replace with your own path
state_shp = state_shp.to_crs('EPSG:4326')
state_shp = state_shp[~state_shp['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]


def generate_bivariate_pallette():
    # define the values of the four corners
    c00 = '#e8e8e8'
    cX0 = '#FADD03'
    c0Y = '#65ACBE'
    cXY = '#C90128'

    C = np.array([to_rgb(c00), to_rgb(cX0), to_rgb(c0Y), to_rgb(cXY)])

    # linear interpolation to generate 5×5 color matrix
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(xs, ys)

    w00 = (1 - X) * (1 - Y)
    wX0 = X * (1 - Y)
    w0Y = (1 - X) * Y
    wXY = X * Y

    colors = (w00[..., None] * C[0]
              + wX0[..., None] * C[1]
              + w0Y[..., None] * C[2]
              + wXY[..., None] * C[3])
    palette = np.array([[to_hex(colors[r, c]) for c in range(5)] for r in range(5)])
    
    return palette


# ========================== calculate uniform tick bounds ==========================
def get_nice_bounds(ds, var_list):
    '''
    ds: xarray.Dataset for county-level characteristics, dimension: (GEOID, year)
    var_list: e.g. ['EH_intensity', 'compound_heat_intensity]
    '''
    # calculate and merge the multi-year mean of each time period
    all_means = []
    for var in var_list:
        m1 = ds[var].sel(year=slice(2006, 2014)).mean(dim='year', skipna=True).values.flatten()
        m2 = ds[var].sel(year=slice(2015, 2023)).mean(dim='year', skipna=True).values.flatten()
        all_means.extend([m1, m2])
    all_data = np.concatenate(all_means)

    # get 5 equal-frequency quantiles
    b = np.nanquantile(all_data, [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # round to the nearest integer: if the second digit is >1, round to the nearest integer; if <1, round to the nearest 0.5
    if b[1] >= 1:
        rounded_b = np.round(b, 0)
    else:
        rounded_b = np.round(b * 2) / 2

    # force adjustment: if the rounding causes overlapping boundaries (e.g. 1.1, 1.3), adjust to maintain monotonic increase
    for i in range(1, len(rounded_b)):
        if rounded_b[i] <= rounded_b[i-1]:
            # if the step size is 0.5, add 0.5, otherwise add 1
            step = 0.5 if b[1] < 1 else 1
            rounded_b[i] = rounded_b[i-1] + step

    return rounded_b


def plot_bivariate_row(ax, gdf, col_x, col_y, bounds_x, bounds_y, palette):
    # col_x / col_y: field name, e.g. 'EH_frequency'
    # bounds_x / bounds_y: palette tick bounds, e.g. [0, 5, 10, 20, 30, 50]

    # map values to 0-4 indices
    idx_x = np.digitize(gdf[col_x], bounds_x[1:-1])
    idx_y = np.digitize(gdf[col_y], bounds_y[1:-1])

    colors = [palette[iy, ix] for ix, iy in zip(idx_x, idx_y)]

    gdf.plot(ax=ax, color=colors, lw=0.1, edgecolor='none')
    state_shp.boundary.plot(ax=ax, edgecolor='lightgray', lw=0.3)

    ax.set_axis_off()


def draw_2d_legend(ax, palette, label_x, label_y, bounds_x, bounds_y):
    n = len(palette)
    for row in range(n):
        for col in range(n):
            ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=palette[row, col], edgecolor='none'))

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(6)); ax.set_yticks(np.arange(6))
    fmt = lambda x: f'{x:.1f}'
    ax.set_xticklabels([fmt(x) for x in bounds_x], fontsize=4)
    ax.set_yticklabels([fmt(y) for y in bounds_y], fontsize=4)
    ax.set_xlabel(label_x, fontsize=6, labelpad=2)
    ax.set_ylabel(label_y, fontsize=6, labelpad=2)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', length=0, pad=1)
    ax.spines[:].set_visible(False)



if __name__ == '__main__':
    combo_path = Path('HI_PM15_lag1')  # replace with your own path
    out_path = Path('output_path')  # replace with your own path

    palette = generate_bivariate_pallette()
    fig = plt.figure(figsize=(14.64 / 2.54, 12 / 2.54))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.3], wspace=0.02, hspace=0.10,
                          left=0.08, right=0.92, top=0.95, bottom=0.05)

    rows_config = {
        'Extreme Heat': ['EH_frequency', 'EH_intensity'],
        'Wildfire Smoke': ['WFS_frequency', 'WFS_intensity'],
        'Compound Heat': ['compound_frequency', 'compound_heat_intensity'],
        'Compound Smoke': ['compound_frequency', 'compound_smoke_intensity']
    }   

    with xr.open_dataset(combo_path / 'identify_county.nc') as county_ds:
        num = 0 # subplot index

        # Period 1 and Period 2 share the same legend
        for row, (event_label, (var1, var2)) in enumerate(rows_config.items()):
            # Frequency axis
            bounds_x = get_nice_bounds(county_ds, [var1])
            if 'Heat' in event_label:
                bounds_y = get_nice_bounds(county_ds, ['EH_intensity', 'compound_heat_intensity'])
            else:
                bounds_y = get_nice_bounds(county_ds, ['WFS_intensity', 'compound_smoke_intensity'])
            
            for col, (start, end) in enumerate([(2006, 2014), (2015, 2023)]):
                ax = fig.add_subplot(gs[row, col])

                df = county_ds.sel(year=slice(start, end)).mean(dim='year', skipna=True).to_dataframe().reset_index()
                gdf = county_shp.merge(df, on='GEOID')
                
                plot_bivariate_row(ax, gdf, var1, var2, bounds_x, bounds_y, palette)

                # subplot title
                if row == 0:
                    ax.set_title(label=f'{start}-{end} Avg', loc='center',
                                 fontsize=8, fontweight='bold', pad=2)
                if col == 0:
                    ax.text(-0.05, 0.5, event_label, va='center', ha='right', rotation='vertical',
                            transform=ax.transAxes, fontsize=8, fontweight='bold')
                
                # subplot index
                if col != 2:
                    ax.text(-0.02, 1.02, f'({chr(num+97)})', va='top', ha='left',
                            transform=ax.transAxes, fontsize=7, fontweight='bold')
                    num += 1
                    
            # third column to plot the corresponding legend
            ax = fig.add_subplot(gs[row, 2])
            # adjust legend position: shift to the right (x0 + 0.05, y0 + 0.01, width, height)
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.05, pos.y0 + 0.01, pos.width, pos.height])

            if 'Heat' in event_label:
                draw_2d_legend(ax, palette, 'Frequency (days)', 'Intensity (°C)',
                               bounds_x, bounds_y)
            elif 'Smoke' in event_label:
                draw_2d_legend(ax, palette, 'Frequency (days)', 'Intensity (μg/m³)',
                               bounds_x, bounds_y)
                
        plt.savefig(out_path / 'Fig1_FreInt_palette.png', dpi=300, bbox_inches='tight')
        plt.close()
