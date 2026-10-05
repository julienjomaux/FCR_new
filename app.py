"""
FCR Capacity Auction Explorer
-----------------------------
Streamlit app giving an overview of the daily FCR (Frequency Containment Reserve)
capacity auction of the FCR Cooperation, based on the public results published
on regelleistung.net.

Data sources (per delivery date):
  * aggregated  -> demand, settlement price and deficit(-)/surplus(+) per country & product
  * anonymous   -> every awarded bid (price, offered MW, allocated MW, country)

Note: the anonymous list only contains *awarded* bids, so the merit order shown is
the accepted part of the bid curve, not the full offered curve.
"""

from __future__ import annotations

import datetime as dt
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# EDITABLE CONSTANTS
# =============================================================================
BASE_URL = "https://www.regelleistung.net/apps/crds/api/v2/tenders/results"
URL_AGGREGATED = (
    BASE_URL + "/aggregated?&productType=FCR&market=CAPACITY&exportFormat=xlsx&deliveryDate={date}"
)
URL_ANONYMOUS = (
    BASE_URL + "/anonymous?&productType=FCR&market=CAPACITY&exportFormat=xlsx&deliveryDate={date}"
)
HTTP_TIMEOUT_S = 60
CACHE_TTL_S = 6 * 3600

# Country name as it appears in the aggregated file -> ISO code used in the bid list
COUNTRY_CODES = {
    "AUSTRIA": "AT",
    "BELGIUM": "BE",
    "DENMARK": "DK",
    "FRANCE": "FR",
    "GERMANY": "DE",
    "NETHERLANDS": "NL",
    "SLOVENIA": "SI",
    "SWITZERLAND": "CH",
    "CZECH_REPUBLIC": "CZ",
}

# Core share (minimum MW that must be procured domestically) and export limit (MW)
# per country. Fill these in from the FCR Cooperation figures published on
# regelleistung.net. Leave None if unknown: the app then skips that constraint.
# Both values can also be edited live in the "Demand & limits" tab.
CORE_SHARE_MW: dict[str, float | None] = {
    "AT": None, "BE": None, "CH": None, "CZ": None, "DE": None,
    "DK": None, "FR": None, "NL": None, "SI": None,
}
EXPORT_LIMIT_MW: dict[str, float | None] = {
    "AT": None, "BE": None, "CH": None, "CZ": None, "DE": None,
    "DK": None, "FR": None, "NL": None, "SI": None,
}

# Fixed colour per country so every chart reads the same way
COUNTRY_COLORS = {
    "DE": "#1f3b73",
    "FR": "#2f7fc1",
    "NL": "#f28e2b",
    "BE": "#e3b505",
    "AT": "#d1495b",
    "CH": "#8c564b",
    "CZ": "#59a14f",
    "DK": "#9c6ade",
    "SI": "#17becf",
}
EXPORT_COLOR = "#2a9d8f"   # surplus  (+)
IMPORT_COLOR = "#e76f51"   # deficit  (-)
PRICE_TOL = 0.005          # EUR/MW tolerance to call two prices "equal"

# =============================================================================
# DATA LOADING
# =============================================================================


def _download(url: str) -> bytes:
    r = requests.get(
        url,
        timeout=HTTP_TIMEOUT_S,
        headers={"User-Agent": "Mozilla/5.0 (FCR auction explorer)"},
    )
    r.raise_for_status()
    if not r.content.startswith(b"PK"):  # xlsx = zip archive
        raise ValueError(
            "The server did not return an Excel file (probably no results for this date yet)."
        )
    return r.content


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def fetch_raw(date_str: str) -> tuple[bytes, bytes]:
    agg = _download(URL_AGGREGATED.format(date=date_str))
    anon = _download(URL_ANONYMOUS.format(date=date_str))
    return agg, anon


def _product_label(p: str) -> str:
    """NEGPOS_00_04 -> 00–04"""
    m = re.search(r"(\d{2})_(\d{2})$", str(p))
    return f"{m.group(1)}–{m.group(2)}" if m else str(p)


@st.cache_data(show_spinner=False)
def parse_aggregated(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long per country/product table, cross-border price per product)."""
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    xb_col = next((c for c in df.columns if c.startswith("CROSSBORDER")), None)
    xb = df[["PRODUCTNAME"]].copy()
    xb["xb_price"] = df[xb_col] if xb_col else np.nan

    rows = []
    for col in df.columns:
        m = re.match(r"^(.*)_DEMAND_\[MW\]$", col)
        if not m:
            continue
        name = m.group(1)
        code = COUNTRY_CODES.get(name, name[:2])
        price_col = f"{name}_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
        surplus_col = next(
            (c for c in df.columns if c.startswith(f"{name}_DEFICIT")), None
        )
        part = pd.DataFrame(
            {
                "product": df["PRODUCTNAME"],
                "country": code,
                "country_name": name.replace("_", " ").title(),
                "demand": pd.to_numeric(df[col], errors="coerce"),
                "price": pd.to_numeric(df.get(price_col), errors="coerce"),
                "surplus": pd.to_numeric(df[surplus_col], errors="coerce")
                if surplus_col
                else np.nan,
            }
        )
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["allocated"] = long["demand"] + long["surplus"]
    long = long.merge(xb, left_on="product", right_on="PRODUCTNAME").drop(columns="PRODUCTNAME")
    long["decoupled"] = (long["price"] - long["xb_price"]).abs() > PRICE_TOL
    long["slot"] = long["product"].map(_product_label)
    # remuneration of BSPs located in the country (EUR, for the 4-h product)
    long["bsp_revenue"] = long["allocated"] * long["price"]
    # what the country's demand costs at its own local price
    long["demand_cost"] = long["demand"] * long["price"]
    return long, xb


@st.cache_data(show_spinner=False)
def parse_anonymous(raw: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "PRODUCT": "product",
        "OFFERED_CAPACITY_PRICE_[EUR/MW]": "bid_price",
        "OFFERED_CAPACITY_[MW]": "offered",
        "ALLOCATED_CAPACITY_[MW]": "allocated",
        "COUNTRY": "country",
        "SETTLEMENTCAPACITY_PRICE_[EUR/MW]": "settlement_price",
        "NOTE": "note",
    }
    df = df.rename(columns=rename)
    for c in ["bid_price", "offered", "allocated", "settlement_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["note"] = df.get("note", pd.Series(index=df.index, dtype=str)).fillna("")
    df["slot"] = df["product"].map(_product_label)
    df["partial"] = (df["allocated"] > 0) & (df["allocated"] < df["offered"])
    return df


# =============================================================================
# CHART HELPERS
# =============================================================================


def color_of(code: str) -> str:
    return COUNTRY_COLORS.get(code, "#888888")


def base_layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    if "title" in kw and isinstance(kw["title"], str):
        kw["title"] = dict(text=kw["title"], y=0.98, yanchor="top", x=0, xanchor="left")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=85, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="closest",
        **kw,
    )
    return fig


def merit_order_fig(
    bids: pd.DataFrame, demand: float | None, title: str, clearing: float | None
) -> go.Figure:
    """Step-style merit order: one bar per bid, width = allocated MW, coloured by country."""
    b = bids[bids["allocated"] > 0].sort_values(["bid_price", "allocated"]).copy()
    b["x_end"] = b["allocated"].cumsum()
    b["x_start"] = b["x_end"] - b["allocated"]
    b["x_mid"] = (b["x_start"] + b["x_end"]) / 2

    fig = go.Figure()
    for code, g in b.groupby("country", sort=False):
        fig.add_trace(
            go.Bar(
                x=g["x_mid"],
                y=g["bid_price"],
                width=g["allocated"],
                name=code,
                marker=dict(color=color_of(code), line=dict(width=0)),
                customdata=np.stack(
                    [g["allocated"], g["offered"], g["x_end"], g["note"], g["settlement_price"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Bid: %{y:.2f} €/MW"
                    "<br>Allocated: %{customdata[0]} MW (offered %{customdata[1]})"
                    "<br>Cumulative: %{customdata[2]} MW"
                    "<br>Settled at: %{customdata[4]:.2f} €/MW"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
    if clearing is not None and not np.isnan(clearing):
        fig.add_hline(
            y=clearing, line=dict(color="#333", dash="dot", width=1),
            annotation_text=f"Cross-border price {clearing:.2f} €/MW",
            annotation_position="top left",
        )
    if demand:
        fig.add_vline(
            x=demand, line=dict(color="#333", dash="dash", width=1),
            annotation_text=f"Demand {demand:.0f} MW", annotation_position="top right",
        )
    fig.update_layout(bargap=0, barmode="overlay")
    fig.update_xaxes(title="Cumulative awarded capacity [MW]", rangemode="tozero")
    fig.update_yaxes(title="Bid price [€/MW]", rangemode="tozero")
    return base_layout(fig, title=title)


def exports_sankey(snap: pd.DataFrame, slot: str) -> go.Figure:
    """Exporters -> FCR pool -> importers for one product (net positions)."""
    exp = snap[snap["surplus"] > 0].sort_values("surplus", ascending=False)
    imp = snap[snap["surplus"] < 0].sort_values("surplus")
    labels = (
        [f"{c} +{v:.0f}" for c, v in zip(exp["country"], exp["surplus"])]
        + ["FCR pool"]
        + [f"{c} {v:.0f}" for c, v in zip(imp["country"], imp["surplus"])]
    )
    colors = [color_of(c) for c in exp["country"]] + ["#bbbbbb"] + [color_of(c) for c in imp["country"]]
    pool = len(exp)
    src, tgt, val, lcol = [], [], [], []
    for i, (c, v) in enumerate(zip(exp["country"], exp["surplus"])):
        src.append(i); tgt.append(pool); val.append(v); lcol.append(color_of(c))
    for j, (c, v) in enumerate(zip(imp["country"], imp["surplus"])):
        src.append(pool); tgt.append(pool + 1 + j); val.append(-v); lcol.append(color_of(c))
    lcol = [_hex_to_rgba(c, 0.35) for c in lcol]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, color=colors, pad=18, thickness=16),
            link=dict(source=src, target=tgt, value=val, color=lcol,
                      hovertemplate="%{value:.0f} MW<extra></extra>"),
        )
    )
    total = exp["surplus"].sum()
    return base_layout(
        fig, height=420,
        title=f"Net exchanges {slot} — {total:.0f} MW flows through the pool "
              "(exporters left, importers right)",
    )


def _hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


# =============================================================================
# APP
# =============================================================================

st.set_page_config(page_title="FCR auction explorer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("FCR capacity auction")
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    delivery = st.date_input(
        "Delivery date",
        value=dt.date.today(),
        min_value=dt.date(2020, 7, 1),
        max_value=tomorrow,
        help="The D-1 auction for tomorrow is usually published around 08:00–09:00 CET.",
    )
    st.caption("Data: regelleistung.net — FCR Cooperation (DE, FR, NL, BE, AT, CH, DK, SI, CZ)")
    with st.expander("Upload files instead"):
        st.caption("If the download fails, upload the two Excel exports manually.")
        up_agg = st.file_uploader("RESULT_OVERVIEW (aggregated)", type="xlsx")
        up_anon = st.file_uploader("RESULT_LIST_ANONYM (anonymous)", type="xlsx")
    if st.button("Clear cache / reload"):
        st.cache_data.clear()

date_str = delivery.strftime("%Y-%m-%d")

# --- load ----------------------------------------------------------------------
if up_agg is not None and up_anon is not None:
    raw_agg, raw_anon = up_agg.getvalue(), up_anon.getvalue()
    source_note = "uploaded files"
else:
    try:
        with st.spinner(f"Downloading FCR results for {date_str}…"):
            raw_agg, raw_anon = fetch_raw(date_str)
        source_note = "regelleistung.net"
    except Exception as e:  # noqa: BLE001
        st.title(f"FCR auction — {date_str}")
        st.error(f"Could not load results for {date_str}: {e}")
        st.info("Pick another date, or upload the two Excel exports in the sidebar.")
        st.stop()

try:
    agg, xb = parse_aggregated(raw_agg)
    bids = parse_anonymous(raw_anon)
except Exception as e:  # noqa: BLE001
    st.error(f"Files downloaded but could not be parsed: {e}")
    st.stop()

if agg.empty:
    st.warning("No results in the file for this date.")
    st.stop()

countries = (
    agg.groupby("country")["demand"].max().sort_values(ascending=False).index.tolist()
)
slots = list(dict.fromkeys(agg["slot"]))

# --- header & KPIs --------------------------------------------------------------
st.title(f"FCR capacity auction — delivery {delivery:%a %d %b %Y}")
st.caption(f"Source: {source_note}. Prices in €/MW per 4-hour product (symmetric NEGPOS).")

total_demand = agg.groupby("slot")["demand"].sum().iloc[0]
xb_prices = xb.set_index(xb["PRODUCTNAME"].map(_product_label))["xb_price"]
n_decoupled = agg.loc[agg["decoupled"], ["slot"]].drop_duplicates().shape[0]
day_cost = agg["demand_cost"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total demand", f"{total_demand:,.0f} MW")
k2.metric("Avg cross-border price", f"{xb_prices.mean():.2f} €/MW")
k3.metric("Min / max price", f"{xb_prices.min():.0f} / {xb_prices.max():.0f} €")
k4.metric("Products with price split", f"{n_decoupled} / {len(slots)}")
k5.metric("Daily procurement cost", f"€{day_cost/1e3:,.0f}k",
          help="Σ demand × local settlement price over the six 4-h products")

(tab_overview, tab_merit, tab_limits, tab_exports, tab_data) = st.tabs(
    ["📊 Results", "📈 Merit order", "🧱 Demand & limits", "🔀 Exports", "🗂 Data"]
)

# =============================================================================
# TAB 1 — RESULTS
# =============================================================================
with tab_overview:
    c1, c2 = st.columns([3, 2])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xb_prices.index, y=xb_prices.values, name="Cross-border",
            mode="lines+markers", line=dict(color="#222", width=3, shape="hv"),
        ))
        for code in countries:
            g = agg[agg["country"] == code]
            if not g["decoupled"].any():
                continue
            fig.add_trace(go.Scatter(
                x=g["slot"], y=g["price"], name=f"{code} (local)",
                mode="lines+markers", line=dict(color=color_of(code), width=2, dash="dot", shape="hv"),
            ))
        fig.update_yaxes(title="€/MW", rangemode="tozero")
        fig.update_xaxes(title="Product (CET)")
        st.plotly_chart(base_layout(fig, title="Settlement prices — cross-border and decoupled countries"),
                        width="stretch")
        if n_decoupled:
            dec = agg[agg["decoupled"]]
            st.info(
                "**Price splits:** "
                + "; ".join(
                    f"{s}: {', '.join(g['country'])} at {g['price'].iloc[0]:.2f} €/MW "
                    f"vs {g['xb_price'].iloc[0]:.2f}"
                    for s, g in dec.groupby("slot")
                )
                + ". A country clearing *below* the cross-border price has hit its export "
                  "limit; *above* means its core share / import limit was binding."
            )
        else:
            st.success("Full price convergence: every country cleared at the cross-border price.")

    with c2:
        piv = agg.pivot(index="country", columns="slot", values="price").loc[countries]
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index, colorscale="Blues",
            text=np.round(piv.values, 2), texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2f} €/MW<extra></extra>",
            colorbar=dict(title="€/MW"),
        ))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, title="Local settlement price per country"),
                        width="stretch")

    # allocated per country stacked
    fig = go.Figure()
    for code in countries:
        g = agg[agg["country"] == code]
        fig.add_trace(go.Bar(x=g["slot"], y=g["allocated"], name=code,
                             marker_color=color_of(code),
                             hovertemplate=f"{code} %{{x}}: %{{y:.0f}} MW<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="MW")
    st.plotly_chart(base_layout(fig, title="Awarded capacity by location of the BSP"),
                    width="stretch")

    summ = agg.groupby(["country", "country_name"]).agg(
        demand=("demand", "first"),
        avg_alloc=("allocated", "mean"),
        avg_price=("price", "mean"),
        avg_surplus=("surplus", "mean"),
        bsp_revenue=("bsp_revenue", "sum"),
        demand_cost=("demand_cost", "sum"),
    ).reset_index().set_index("country").loc[countries].reset_index()
    st.dataframe(
        summ.rename(columns={
            "country": "Country", "country_name": "Name", "demand": "Demand [MW]",
            "avg_alloc": "Avg awarded [MW]", "avg_price": "Avg price [€/MW]",
            "avg_surplus": "Avg net export [MW]", "bsp_revenue": "BSP revenue [€]",
            "demand_cost": "Demand cost [€]",
        }),
        hide_index=True, width="stretch",
        column_config={
            "Avg awarded [MW]": st.column_config.NumberColumn(format="%.0f"),
            "Avg price [€/MW]": st.column_config.NumberColumn(format="%.2f"),
            "Avg net export [MW]": st.column_config.NumberColumn(format="%+.0f"),
            "BSP revenue [€]": st.column_config.NumberColumn(format="%,.0f"),
            "Demand cost [€]": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

# =============================================================================
# TAB 2 — MERIT ORDER
# =============================================================================
with tab_merit:
    st.caption(
        "Built from the anonymous bid list, which only contains **awarded** bids. "
        "Indivisible bids and partially accepted bids are flagged in the tooltip and table."
    )
    m1, m2 = st.columns([1, 3])
    with m1:
        slot = st.radio("Product", slots, horizontal=False, key="mo_slot")
        view = st.radio("View", ["All countries together", "Per country"], key="mo_view")
    bslot = bids[bids["slot"] == slot]
    snap = agg[agg["slot"] == slot].set_index("country")
    with m2:
        if view == "All countries together":
            fig = merit_order_fig(
                bslot, snap["demand"].sum(),
                f"Cooperation-wide merit order — {slot}",
                xb_prices.get(slot),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            sel = st.multiselect("Countries", countries, default=countries[:4], key="mo_cty")
            cols = st.columns(2)
            for i, code in enumerate(sel):
                g = bslot[bslot["country"] == code]
                if g.empty:
                    cols[i % 2].info(f"{code}: no awarded bids in {slot}")
                    continue
                f = merit_order_fig(
                    g, snap.loc[code, "demand"] if code in snap.index else None,
                    f"{code} — {slot} · local price {snap.loc[code, 'price']:.2f} €/MW",
                    snap.loc[code, "price"] if code in snap.index else None,
                )
                f.update_layout(height=340, showlegend=False)
                cols[i % 2].plotly_chart(f, width="stretch")

    # bid statistics for the product
    st.subheader(f"Bid statistics — {slot}")
    stats = bslot.groupby("country").agg(
        bids=("allocated", "size"),
        awarded_mw=("allocated", "sum"),
        min_price=("bid_price", "min"),
        wavg_price=("bid_price", lambda s: np.average(s, weights=bslot.loc[s.index, "allocated"])
                    if bslot.loc[s.index, "allocated"].sum() > 0 else np.nan),
        max_price=("bid_price", "max"),
        zero_price_mw=("allocated", lambda s: s[bslot.loc[s.index, "bid_price"] <= 0].sum()),
        indivisible=("note", lambda s: (s == "INDIVISIBLE").sum()),
        partial=("partial", "sum"),
    ).reset_index().sort_values("awarded_mw", ascending=False)
    st.dataframe(
        stats.rename(columns={
            "country": "Country", "bids": "# bids", "awarded_mw": "Awarded [MW]",
            "min_price": "Min bid", "wavg_price": "Wavg bid", "max_price": "Max (marginal) bid",
            "zero_price_mw": "MW bid at ≤0 €", "indivisible": "# indivisible",
            "partial": "# partially accepted",
        }),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ["Min bid", "Wavg bid", "Max (marginal) bid"]},
    )

# =============================================================================
# TAB 3 — DEMAND & LIMITS
# =============================================================================
with tab_limits:
    st.markdown(
        "Each country's **demand** must be covered by the cooperation. The **core share** "
        "is the minimum that has to be procured from BSPs inside the country, so the maximum "
        "import is *demand − core*. The **export limit** caps how much a country's BSPs "
        "can deliver to the others. Demand comes from the auction file; core share and export "
        "limit are taken from the constants in the code and can be edited below."
    )
    dem = agg.groupby("country")["demand"].first().loc[countries]
    base_tbl = pd.DataFrame({
        "Country": countries,
        "Demand [MW]": dem.values,
        "Core share [MW]": [CORE_SHARE_MW.get(c) for c in countries],
        "Export limit [MW]": [EXPORT_LIMIT_MW.get(c) for c in countries],
    })
    edited = st.data_editor(
        base_tbl, hide_index=True, width="stretch", key=f"limits_{date_str}",
        disabled=["Country", "Demand [MW]"],
        column_config={
            "Core share [MW]": st.column_config.NumberColumn(min_value=0, step=1),
            "Export limit [MW]": st.column_config.NumberColumn(min_value=0, step=1),
        },
    )
    lim = edited.set_index("Country")
    lim["Max import [MW]"] = lim["Demand [MW]"] - lim["Core share [MW]"]

    # infer binding export limits from price splits (country below cross-border price)
    lo = agg[agg["decoupled"] & (agg["price"] < agg["xb_price"])]
    if not lo.empty:
        st.caption(
            "Observed binding export positions (local price < cross-border price): "
            + "; ".join(f"{r.country} {r.slot}: {r.surplus:+.0f} MW"
                        for r in lo.itertuples())
            + ". When several countries decouple together, the limit applies to the group."
        )

    st.subheader("Position of each country vs its constraints")
    l1, l2 = st.columns([1, 3])
    with l1:
        lim_slot = st.radio("Product", slots, key="lim_slot")
    snap = agg[agg["slot"] == lim_slot].set_index("country").loc[countries]
    with l2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=countries, x=snap["allocated"], orientation="h", name="Awarded in country",
            marker_color=[color_of(c) for c in countries],
            hovertemplate="%{y}: %{x:.0f} MW awarded<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            y=countries, x=snap["demand"], mode="markers", name="Demand",
            marker=dict(symbol="line-ns", size=26, line=dict(width=3, color="#111")),
            hovertemplate="%{y}: demand %{x:.0f} MW<extra></extra>",
        ))
        core = lim["Core share [MW]"].reindex(countries)
        if core.notna().any():
            fig.add_trace(go.Scatter(
                y=countries, x=core, mode="markers", name="Core share (min. domestic)",
                marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=IMPORT_COLOR)),
                hovertemplate="%{y}: core %{x:.0f} MW<extra></extra>",
            ))
        el = lim["Export limit [MW]"].reindex(countries)
        if el.notna().any():
            fig.add_trace(go.Scatter(
                y=countries, x=snap["demand"] + el, mode="markers",
                name="Demand + export limit (max domestic)",
                marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=EXPORT_COLOR)),
                hovertemplate="%{y}: max %{x:.0f} MW<extra></extra>",
            ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="MW")
        st.plotly_chart(base_layout(fig, height=440, title=f"Awarded vs demand — {lim_slot}"),
                        width="stretch")

    # utilisation table across products
    util = agg[["country", "slot", "surplus"]].merge(
        lim[["Export limit [MW]", "Max import [MW]"]], left_on="country", right_index=True
    )
    util["use"] = np.where(
        util["surplus"] >= 0,
        util["surplus"] / util["Export limit [MW]"],
        util["surplus"] / util["Max import [MW]"],
    )
    if util["use"].notna().any():
        piv = util.pivot(index="country", columns="slot", values="use").loc[countries] * 100
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index, zmid=0, zmin=-100, zmax=100,
            colorscale=[[0, IMPORT_COLOR], [0.5, "#f7f7f7"], [1, EXPORT_COLOR]],
            text=np.round(piv.values, 0), texttemplate="%{text}%",
            hovertemplate="%{y} %{x}: %{z:.0f}%<extra></extra>",
            colorbar=dict(title="% of limit"),
        ))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(
            fig, height=380,
            title="Limit utilisation: + export / export limit, − import / (demand − core)"),
            width="stretch")
    else:
        st.info("Fill in core share and export limits above to see limit utilisation.")

    st.dataframe(lim.reset_index(), hide_index=True, width="stretch")

# =============================================================================
# TAB 4 — EXPORTS
# =============================================================================
with tab_exports:
    st.caption("Net position = awarded in country − demand. Positive = exporter, negative = importer.")

    piv = agg.pivot(index="country", columns="slot", values="surplus").loc[countries]
    vmax = float(np.nanmax(np.abs(piv.values))) or 1
    e1, e2 = st.columns([2, 3])
    with e1:
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index, zmid=0, zmin=-vmax, zmax=vmax,
            colorscale=[[0, IMPORT_COLOR], [0.5, "#f7f7f7"], [1, EXPORT_COLOR]],
            text=piv.values, texttemplate="%{text:+.0f}",
            hovertemplate="%{y} %{x}: %{z:+.0f} MW<extra></extra>",
            colorbar=dict(title="MW"),
        ))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, height=440, title="Net export per country and product [MW]"),
                        width="stretch")
    with e2:
        fig = go.Figure()
        for code in countries:
            g = agg[agg["country"] == code]
            fig.add_trace(go.Bar(
                x=g["slot"], y=g["surplus"], name=code, marker_color=color_of(code),
                hovertemplate=f"{code} %{{x}}: %{{y:+.0f}} MW<extra></extra>",
            ))
        fig.update_layout(barmode="relative")
        fig.add_hline(y=0, line=dict(color="#333", width=1))
        fig.update_yaxes(title="MW  (↑ export / ↓ import)")
        st.plotly_chart(base_layout(fig, height=440, title="Exporters above, importers below"),
                        width="stretch")

    ex_slot = st.select_slider("Product for the flow diagram", options=slots, key="ex_slot")
    st.plotly_chart(exports_sankey(agg[agg["slot"] == ex_slot], ex_slot), width="stretch")

    # daily view: avg net position & self-sufficiency
    daily = agg.groupby("country").agg(
        demand=("demand", "first"), alloc=("allocated", "mean"), surplus=("surplus", "mean"),
        min_s=("surplus", "min"), max_s=("surplus", "max"),
    ).loc[countries]
    daily["self_suff"] = daily["alloc"] / daily["demand"] * 100
    fig = go.Figure(go.Bar(
        x=daily.index, y=daily["self_suff"],
        marker_color=[EXPORT_COLOR if v >= 100 else IMPORT_COLOR for v in daily["self_suff"]],
        text=[f"{v:.0f}%" for v in daily["self_suff"]], textposition="outside",
        customdata=np.stack([daily["min_s"], daily["max_s"]], axis=-1),
        hovertemplate="%{x}: %{y:.0f}% of demand awarded domestically"
                      "<br>Net position range %{customdata[0]:+.0f} … %{customdata[1]:+.0f} MW<extra></extra>",
    ))
    fig.add_hline(y=100, line=dict(color="#333", dash="dash", width=1))
    fig.update_yaxes(title="% of own demand", rangemode="tozero")
    st.plotly_chart(base_layout(fig, height=360,
                                title="Daily average self-sufficiency (awarded domestically / demand)"),
                    width="stretch")

# =============================================================================
# TAB 5 — DATA
# =============================================================================
with tab_data:
    st.subheader("Aggregated results (long format)")
    st.dataframe(agg, hide_index=True, width="stretch")
    st.download_button("Download aggregated (CSV)", agg.to_csv(index=False).encode(),
                       f"fcr_aggregated_{date_str}.csv", "text/csv")
    st.subheader("Awarded bids")
    st.dataframe(bids, hide_index=True, width="stretch")
    st.download_button("Download bids (CSV)", bids.to_csv(index=False).encode(),
                       f"fcr_bids_{date_str}.csv", "text/csv")
    st.download_button("Original aggregated xlsx", raw_agg,
                       f"RESULT_OVERVIEW_FCR_{date_str}.xlsx")
    st.download_button("Original anonymous xlsx", raw_anon,
                       f"RESULT_LIST_ANONYM_FCR_{date_str}.xlsx")
