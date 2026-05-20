#!/usr/bin/env python3
import os, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs

plt.close("all")

# =========================
# USER SETTINGS
# =========================
PATTERN = "stat_trs_scl_pos.addwind850_addwind10m_addmslp_addprec_1.nc"

ROOT_HIST = "YOUR PATH"
ROOT_SSP  = "YOUR PATH"

NBOOT = 2000
ALPHA = 0.05
SEED  = 0

# If you want to mask intensity where track density is low
MASK_INTENSITY_WHEN_TRD_LT = None  # set None to disable

# Earth radius used in your plotting
R_EARTH = 6.371e6  # m

# =========================
# HELPERS
# =========================
import os, glob
import xarray as xr

def open_ensemble(root, pattern, member_dim="member"):
    """
    Open an ensemble of NetCDFs located under:
      root/*/*/TOTAL*/stat_old/<pattern>

    pattern: e.g. "stat_trs_scl_pos.addwind850_addwind10m_addmslp_addprec*_1.nc"
    """
    # TOTAL sometimes has suffixes, so use TOTAL*
    search = os.path.join(root, "*", "*", "TOTAL*", "stat_old", pattern)
    files = sorted(glob.glob(search))

    print("Search pattern:", search)
    print(f"[{os.path.basename(root)}] Found files:", len(files))
    if files:
        print("Example:", files[0])

    if not files:
        raise SystemExit(f"No files found with: {search}")

    return xr.open_mfdataset(files, concat_dim=member_dim, combine="nested")

def bootstrap_sig_mask_paired(ssp_da, hist_da, member_dim="member",
                              nboot=2000, alpha=0.05, seed=0):
    """
    Paired bootstrap on ensemble members.
    Returns:
      dbar   : mean change (ssp - hist)
      lo, hi : bootstrap CI bounds
      mask   : True where CI excludes 0 (stipple)
    """
    rng = np.random.default_rng(seed)

    d = (ssp_da - hist_da)  # (member, y, x) or (member, lat, lon)
    N = d.sizes[member_dim]
    if N < 2:
        raise ValueError("Need at least 2 ensemble members for bootstrap.")

    # stack all non-member dims into a single 'points' axis for speed
    other_dims = [dd for dd in d.dims if dd != member_dim]
    d2 = d.transpose(member_dim, *other_dims).stack(points=other_dims)  # (member, points)

    arr = d2.values  # numpy array (member, points)
    # bootstrap indices (nboot, N)
    idx = rng.integers(0, N, size=(nboot, N))

    # bootstrap mean changes (nboot, points)
    boot_means = arr[idx, :].mean(axis=1)

    # percentile CI
    lo = np.percentile(boot_means, 100 * (alpha/2), axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha/2), axis=0)

    # significant if CI excludes 0
    mask = (lo > 0) | (hi < 0)

    # unstack back
    lo_da   = xr.DataArray(lo,  coords=d2.points.coords).unstack("points")
    hi_da   = xr.DataArray(hi,  coords=d2.points.coords).unstack("points")
    mask_da = xr.DataArray(mask, coords=d2.points.coords).unstack("points")
    dbar_da = d.mean(member_dim)

    return dbar_da, lo_da, hi_da, mask_da

def add_polar_map_format(ax, circle):
    ax.set_extent([0, 359.99, 20, 90], ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 xlocs=np.arange(-180, 180, 45),
                 ylocs=np.arange(0, 90, 30))

def stipple_mask(ax, X, Y, mask, step=3, s=2):
    """
    Plot stippling where mask==True.
    Works whether X,Y are 1D (axes) or 2D (meshgrid).
    mask must be 2D with shape (ny, nx).
    """
    m = np.asarray(mask).astype(bool)

    # Subsample mask
    m_sub = m[::step, ::step]
    yy, xx = np.where(m_sub)

    # Handle 1D vs 2D coordinates
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim == 1 and Y.ndim == 1:
        # X: (nx,), Y: (ny,)
        xpts = X[::step][xx]
        ypts = Y[::step][yy]
    else:
        # X,Y are 2D grids (ny,nx)
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]
        xpts = X_sub[yy, xx]
        ypts = Y_sub[yy, xx]

    ax.scatter(xpts, ypts, s=s, marker=".", linewidths=0, color="black",
               transform=ccrs.NorthPolarStereo(), zorder=5)

# =========================
# LOAD ENSEMBLES
# =========================
dsH = open_ensemble(ROOT_HIST, PATTERN, member_dim="member")
dsS = open_ensemble(ROOT_SSP,  PATTERN, member_dim="member")

# sanity: same number of members
if dsH.dims.get("member", None) != dsS.dims.get("member", None):
    raise ValueError(f"Different member counts: HIST={dsH.dims.get('member')} SSP={dsS.dims.get('member')}")

# coordinates for plotting
X = dsH["X-pos"].values * R_EARTH
Y = dsH["Y-pos"].values * R_EARTH

# =========================
# FIELDS
# =========================
trdH = dsH["tden"] / 8.0  # as in your plot (per month)
trdS = dsS["tden"] / 8.0

mstrH = dsH["mstr"]
mstrS = dsS["mstr"]

# means
trdH_mean  = trdH.mean("member")
mstrH_mean = mstrH.mean("member")

# bootstrap significance for differences
dtrd_mean,  dtrd_lo,  dtrd_hi,  dtrd_sig  = bootstrap_sig_mask_paired(trdS,  trdH,  nboot=NBOOT, alpha=ALPHA, seed=SEED)
dmstr_mean, dmstr_lo, dmstr_hi, dmstr_sig = bootstrap_sig_mask_paired(mstrS, mstrH, nboot=NBOOT, alpha=ALPHA, seed=SEED)

# optional intensity masking based on HIST mean track density
if MASK_INTENSITY_WHEN_TRD_LT is not None:
    msk = trdH_mean.values * 12.0 < MASK_INTENSITY_WHEN_TRD_LT  # compare to original raw threshold
    mstrH_plot  = np.ma.masked_where(msk, mstrH_mean.values)
    dmstr_plot  = np.ma.masked_where(msk, dmstr_mean.values)
    dmstr_sig_plot = np.ma.masked_where(msk, dmstr_sig.values.astype(bool))
else:
    mstrH_plot = mstrH_mean.values
    dmstr_plot = dmstr_mean.values
    dmstr_sig_plot = dmstr_sig.values.astype(bool)

# =========================
# PLOTTING
# =========================
cmap_main = plt.cm.viridis
cmap_diff = plt.cm.RdBu_r

fig, axs = plt.subplots(
    2, 2,
    subplot_kw={"projection": ccrs.NorthPolarStereo()},
    figsize=(13, 12)
)

# circular boundary
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# Fixed levels
levels_trd   = np.arange(0, 22, 2)
ticks_trd    = np.arange(0, 21, 4)
levels_mstr  = np.arange(1, 7.1, .5)
ticks_mstr  = np.arange(1, 8, 1)

# --- Panel (0,0): HIST track density ---
ax = axs[0, 0]
add_polar_map_format(ax, circle)

cs = ax.contourf(
    X, Y, trdH_mean.values,
    levels=levels_trd,
    cmap=cmap_main,
    extend="max",
    transform=ccrs.NorthPolarStereo()
)

cbar = fig.colorbar(
    cs, ax=ax, orientation="horizontal",
    fraction=0.046, pad=0.08, ticks=ticks_trd
)
cbar.set_label(r"Mean Track density [$10^{-6}km^{-2}month^{-1}$]", fontsize=11)
cbar.ax.tick_params(labelsize=12) 

# --- Panel (0,1): HIST mean intensity ---
ax = axs[0, 1]
add_polar_map_format(ax, circle)

cs = ax.contourf(
    X, Y, mstrH_plot,
    levels=levels_mstr,
    cmap=cmap_main,
    extend="both",
    transform=ccrs.NorthPolarStereo()
)

cbar = fig.colorbar(
    cs, ax=ax, orientation="horizontal",
    fraction=0.046, pad=0.08, ticks=ticks_mstr
)
cbar.set_label(r"Mean intensity [$10^{-5} s^{-1}$] ", fontsize=11)
cbar.ax.tick_params(labelsize=12) 

# --- Panel (1,0): SSP370 − HIST track density ---
ax = axs[1, 0]
add_polar_map_format(ax, circle)

mx = float(np.nanmax(np.abs(dtrd_mean.values)))
levels_diff = np.arange(-2.0, 2.1, 0.25)
ticks_trd_diff = np.arange(-2.0, 2.1, 0.5)

cs = ax.contourf(
    X, Y, dtrd_mean.values,
    levels=levels_diff,
    cmap=cmap_diff,
    extend="both",
    transform=ccrs.NorthPolarStereo()
)

cbar = fig.colorbar(
    cs, ax=ax, orientation="horizontal",
    fraction=0.046, pad=0.08, ticks = ticks_trd_diff
)
cbar.set_label(r"SSP370 − HIST Mean Track density [10$^{-6}$km$^{-2}$month$^{-1}$]", fontsize=12)
cbar.ax.tick_params(labelsize=12) 

stipple_mask(ax, X, Y, dtrd_sig.values.astype(bool), step=2, s=3)

# --- Panel (1,1): SSP370 − HIST mean intensity ---
ax = axs[1, 1]
add_polar_map_format(ax, circle)

mx = float(np.nanmax(np.abs(dmstr_plot)))

levels_diff = np.arange(-1.0, 1.1, 0.25)
ticks_mstr_diff = np.arange(-1.0, 1.1, 0.5)

cs = ax.contourf(
    X, Y, dmstr_plot,
    levels=levels_diff,
    cmap=cmap_diff,
    extend="both",
    transform=ccrs.NorthPolarStereo()
)

cbar = fig.colorbar(
    cs, ax=ax, orientation="horizontal",
    fraction=0.046, pad=0.08, ticks = ticks_mstr_diff
)
cbar.set_label(r"SSP370 − HIST Mean Intensity [$10^{-5} s^{-1}$]", fontsize=12)
cbar.ax.tick_params(labelsize=12) 

stipple_mask(ax, X, Y, dmstr_sig_plot, step=2, s=3)

plt.tight_layout()
plt.show()

fig.savefig(
    "tracks_density_intensity_hist_and_change_bootstrap_stipple_all.png",
    dpi=150,
    bbox_inches="tight"
)
