# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:annorxiver]
#     language: python
#     name: conda-env-annorxiver-py
# ---

# # Figures for Corpora Comparison between bioRxiv,  Pubmed Central, New York Times

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

from cairosvg import svg2png
from IPython.display import Image
import plotnine as p9

from annorxiver_modules.corpora_comparison_helper import (
    calculate_confidence_intervals,
    create_lemma_count_df,
    plot_bargraph,
    plot_point_bar_figure,
)
# -


subset = 20

# # KL Divergence Graph

kl_divergence_df = pd.read_csv(
    "output/comparison_stats/corpora_kl_divergence.tsv", sep="\t"
)
kl_divergence_df.head()

g = (
    p9.ggplot(
        kl_divergence_df.replace(
            {
                "biorxiv_vs_pmc": "bioRxiv-PMC",
                "biorxiv_vs_nytac": "bioRxiv-NYTAC",
                "pmc_vs_nytac": "PMC-NYTAC",
            }
        ).rename(index=str, columns={"comparison": "Comparison"})
    )
    + p9.aes(
        x="factor(num_terms)",
        y="KL_divergence",
        fill="Comparison",
        color="Comparison",
        group="Comparison",
    )
    + p9.geom_point(size=3)
    + p9.geom_line(linetype="dashed")
    + p9.scale_fill_brewer(type="qual", palette="Paired", direction=-1)
    + p9.scale_color_brewer(
        type="qual",
        palette="Paired",
        direction=-1,
    )
    + p9.labs(
        x="Number of terms evaluated",
        y="Kullback–Leibler Divergence",
    )
    + p9.theme_seaborn(
        context="paper",
        style="ticks",
        font_scale=2,
    )
    + p9.theme(figure_size=(10, 6), text=p9.element_text(family="Arial"))
)
g.save("output/svg_files/corpora_kl_divergence.svg")
g.save("output/figures/corpora_kl_divergence.png", dpi=500)
print(g)

# # bioRxiv vs Pubmed Central

full_text_comparison = pd.read_csv(
    "output/comparison_stats/biorxiv_vs_pmc_comparison.tsv", sep="\t"
)
full_text_comparison.head()

# ## Line Plots

full_plot_df = calculate_confidence_intervals(full_text_comparison)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df.sort_values("odds_ratio", ascending=False).iloc[:-2].tail(subset)
    )
    .replace("rna", "RNA")
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x)),
    )
)
plot_df.head()

g = (
    p9.ggplot(
        plot_df, p9.aes(y="lemma", x="lower_odds", xend="upper_odds", yend="lemma")
    )
    + p9.geom_segment(color="#253494", size=6, alpha=0.7)
    + p9.scale_y_discrete(
        limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    )
    + p9.scale_x_continuous(limits=(-3, 3))
    + p9.geom_vline(p9.aes(xintercept=0), linetype="--", color="grey")
    + p9.annotate(
        "segment",
        x=0.5,
        xend=2.5,
        y=1.5,
        yend=1.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="bioRxiv Enriched", x=1.5, y=2.5, size=14, alpha=0.7)
    + p9.annotate(
        "segment",
        x=-0.5,
        xend=-2.5,
        y=39.5,
        yend=39.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="PMC Enriched", x=-1.5, y=38.5, size=14, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.8, font="Arial")
    + p9.theme(
        figure_size=(11, 8.5),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="bioRxiv vs PMC log2(Odds Ratio)")
)
g.save("output/svg_files/biorxiv_pmc_frequency_odds.svg")
g.save("output/svg_files/biorxiv_pmc_frequency_odds.png")
print(g)

count_plot_df = (
    create_lemma_count_df(plot_df, "bioRxiv", "pmc")
    .replace({"pmc": "PMC"})
    .assign(
        repository=lambda x: pd.Categorical(
            x.repository.tolist(), categories=["bioRxiv", "PMC"]
        )
    )
)
count_plot_df.head()

g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_pmc_frequency_bar.svg")
print(g)

# +
fig_output_path = "output/figures/biorxiv_vs_pubmed_central.png"

fig = plot_point_bar_figure(
    "output/svg_files/biorxiv_pmc_frequency_odds.svg",
    "output/svg_files/biorxiv_pmc_frequency_bar.svg",
)

# save generated SVG files
svg2png(bytestring=fig.to_str(), write_to=fig_output_path, dpi=600)

Image(fig_output_path)
# -

# # bioRxiv vs Reference

full_text_comparison = pd.read_csv(
    "output/comparison_stats/biorxiv_nytac_comparison.tsv", sep="\t"
)
full_text_comparison.head()

# ## Line Plots

full_plot_df = calculate_confidence_intervals(full_text_comparison)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df.sort_values("odds_ratio", ascending=False).iloc[:-2].tail(subset)
    )
    .replace("rna", "RNA")
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x)),
    )
)
plot_df.head()

# +
g = (
    p9.ggplot(
        plot_df, p9.aes(y="lemma", x="lower_odds", xend="upper_odds", yend="lemma")
    )
    + p9.geom_segment(color="#253494", size=6, alpha=0.7)
    + p9.scale_y_discrete(
        limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    )
    + p9.geom_vline(p9.aes(xintercept=0), linetype="--", color="grey")
    + p9.annotate(
        "segment",
        x=5,
        xend=17,
        y=1.5,
        yend=1.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="bioRxiv Enriched", x=9, y=2.5, size=14, alpha=0.7)
    + p9.annotate(
        "segment",
        x=-5,
        xend=-17,
        y=39.5,
        yend=39.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="NYTAC Enriched", x=-9, y=38.5, size=14, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.8, font="Arial")
    + p9.theme(
        figure_size=(11, 8.5),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="bioRxiv vs NYTAC log2(Odds Ratio)")
)

g.save("output/svg_files/biorxiv_nytac_frequency_odds.svg")
g.save("output/svg_files/biorxiv_nytac_frequency_odds.png", dpi=250)
print(g)
# -

count_plot_df = create_lemma_count_df(plot_df, "bioRxiv", "NYTAC").assign(
    repository=lambda x: pd.Categorical(
        x.repository.tolist(), categories=["bioRxiv", "NYTAC"]
    )
)
count_plot_df.head()

g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_nytac_frequency_bar.svg")
print(g)

# +
fig_output_path = "output/figures/biorxiv_vs_reference.png"

fig = plot_point_bar_figure(
    "output/svg_files/biorxiv_nytac_frequency_odds.svg",
    "output/svg_files/biorxiv_nytac_frequency_bar.svg",
)

# save generated SVG files
svg2png(bytestring=fig.to_str(), write_to=fig_output_path, dpi=75)

Image(fig_output_path)
# -

# # PMC vs Reference

full_text_comparison = pd.read_csv(
    "output/comparison_stats/pmc_nytac_comparison.tsv", sep="\t"
)
full_text_comparison.head()

# ## Line Plots

full_plot_df = calculate_confidence_intervals(full_text_comparison)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=False)
    .drop([17, 154])
    .head(subset)
    .append(
        full_plot_df.sort_values("odds_ratio", ascending=False).iloc[:-2].tail(subset)
    )
    .replace("rna", "RNA")
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x)),
    )
)
plot_df.head()

g = (
    p9.ggplot(
        plot_df, p9.aes(y="lemma", x="lower_odds", xend="upper_odds", yend="lemma")
    )
    + p9.geom_segment(color="#253494", size=6, alpha=0.7)
    + p9.scale_y_discrete(
        limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    )
    + p9.geom_vline(p9.aes(xintercept=0), linetype="--", color="grey")
    + p9.annotate(
        "segment",
        x=5,
        xend=17,
        y=1.5,
        yend=1.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="PMC Enriched", x=9, y=2.5, size=14, alpha=0.7)
    + p9.annotate(
        "segment",
        x=-5,
        xend=-17,
        y=39.5,
        yend=39.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="NYTAC Enriched", x=-9, y=38.5, size=14, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.8, font="Arial")
    + p9.theme(
        figure_size=(11, 8.5),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="PMC vs NYTAC log2(Odds Ratio)")
)
g.save("output/svg_files/pmc_nytac_frequency_odds.svg")
g.save("output/svg_files/pmc_nytac_frequency_odds.png", dpi=250)
print(g)

count_plot_df = create_lemma_count_df(plot_df, "pmc", "reference").replace(
    {"pmc": "PMC", "reference": "NYTAC"}
)
count_plot_df.head()

g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/pmc_nytac_frequency_bar.svg", dpi=75)
print(g)

# +
fig_output_path = "output/figures/pmc_vs_reference.png"

fig = plot_point_bar_figure(
    "output/svg_files/pmc_nytac_frequency_odds.svg",
    "output/svg_files/pmc_nytac_frequency_bar.svg",
)

# save generated SVG files
svg2png(bytestring=fig.to_str(), write_to=fig_output_path, dpi=75)

Image(fig_output_path)
# -

# # Preprint vs Published

preprint_published_comparison = pd.read_csv(
    "output/comparison_stats/preprint_to_published_comparison.tsv", sep="\t"
).assign(odds_ratio=lambda x: 1 / x.odds_ratio.values)
preprint_published_comparison.head()

# ## Line Plot

full_plot_df = calculate_confidence_intervals(preprint_published_comparison)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=False)
    .iloc[3:]
    .head(subset)
    .append(full_plot_df.sort_values("odds_ratio", ascending=False).tail(subset))
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x)),
    )
)
plot_df.head()

g = (
    p9.ggplot(
        plot_df, p9.aes(y="lemma", x="lower_odds", xend="upper_odds", yend="lemma")
    )
    + p9.geom_segment(color="#253494", size=6, alpha=0.7)
    + p9.scale_y_discrete(
        limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    )
    + p9.scale_x_continuous(limits=(-3, 3))
    + p9.geom_vline(p9.aes(xintercept=0), linetype="--", color="grey")
    + p9.annotate(
        "segment",
        x=0.5,
        xend=2.5,
        y=1.5,
        yend=1.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="Published  Enriched", x=1.5, y=2.5, size=14, alpha=0.7)
    + p9.annotate(
        "segment",
        x=-0.5,
        xend=-2.5,
        y=39.5,
        yend=39.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="Preprint Enriched", x=-1.5, y=38.5, size=14, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.7, font="Arial")
    + p9.theme(
        figure_size=(11, 8.5),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="Preprint vs Published log2(Odds Ratio)")
)
g.save("output/svg_files/preprint_published_frequency_odds.svg")
g.save("output/svg_files/preprint_published_frequency_odds.png")
print(g)

count_plot_df = create_lemma_count_df(plot_df, "preprint", "published").replace(
    {"preprint": "Preprint", "published": "Published"}
)
count_plot_df.head()

g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/preprint_published_frequency_bar.svg")
print(g)

# +
fig_output_path = "output/figures/preprint_published_comparison.png"

fig = plot_point_bar_figure(
    "output/svg_files/preprint_published_frequency_odds.svg",
    "output/svg_files/preprint_published_frequency_bar.svg",
)

# save generated SVG files
svg2png(bytestring=fig.to_str(), write_to=fig_output_path, dpi=600)

Image(fig_output_path)
