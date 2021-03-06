# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Figure Generation for Manuscript

# +
from pathlib import Path
from IPython.display import Image, display, SVG

from cairosvg import svg2png
from lxml import etree
import numpy as np
from svgutils.compose import Unit
import svgutils.transform as sg

from annorxiver_modules.pca_plot_helper import load_clouds
# -

# ## Figure 1

corpora_comparison_path = Path("../biorxiv/corpora_comparison/output/")

# +
panel_one = sg.fromfile(corpora_comparison_path / "svg_files/corpora_kl_divergence.svg")

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(30, 30)

# +
panel_two = sg.fromfile(
    corpora_comparison_path / "svg_files/biorxiv_pmc_frequency_odds.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(40, 598)

# +
panel_three = sg.fromfile(
    corpora_comparison_path / "svg_files/biorxiv_pmc_frequency_bar.svg"
)

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(820, 580)

# +
panel_four = sg.fromfile(
    corpora_comparison_path / "svg_files/preprint_published_frequency_odds.svg"
)

panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(20, 1018)

# +
panel_five = sg.fromfile(
    corpora_comparison_path / "svg_files/preprint_published_frequency_bar.svg"
)

panel_five_size = (
    np.round(float(panel_five.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_five.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale(x=scale_x, y=scale_y)
panel_five.moveto(800, 1000)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(10, 600, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(800, 600, "C", size=22, weight="bold")
panel_four_label = sg.TextElement(30, 1010, "D", size=22, weight="bold")
panel_five_label = sg.TextElement(800, 1010, "E", size=22, weight="bold")

# +
figure_one = sg.SVGFigure(
    Unit(
        max(
            [
                panel_one_size[0],
                panel_two_size[0] + panel_three_size[0],
                panel_four_size[0] + panel_five_size[0],
            ]
        )
        - 100
    ),
    Unit(
        panel_one_size[1]
        + max(panel_two_size[1], panel_three_size[1])
        + max(panel_four_size[1], panel_five_size[1])
        - 150,
    ),
)

figure_one.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_five,
        panel_one_label,
        panel_two_label,
        panel_three_label,
        panel_four_label,
        panel_five_label,
    ]
)
display(SVG(figure_one.to_str()))
# -

# save generated SVG files
figure_one.save("output/figure_one_panels.svg")

# ## Figure 2

word_association_path = Path(
    "../biorxiv/pca_association_experiment/output/word_pca_similarity/figure_pieces"
)
pca_association_path = Path(
    "../biorxiv/pca_association_experiment/output/pca_plots/svg_files"
)

# +
panel_one = sg.fromfile(
    pca_association_path / "scatterplot_files/pca01_v_pca02_reversed.svg"
)

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.2
scale_y = 1.2

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(50, 20)

# +
panel_two = load_clouds(
    str(word_association_path / "pca_01_cossim_word_cloud.png"), figure_size=(10, 7)
)

panel_two_size = (
    float(panel_two.root.attrib["width"]),
    float(panel_two.root.attrib["height"]),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(50, 520)

# +
panel_three = load_clouds(
    str(word_association_path / "pca_02_cossim_word_cloud.png"), figure_size=(10, 7)
)

panel_three_size = (
    float(panel_three.root.attrib["width"]),
    float(panel_three.root.attrib["height"]),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(660, 520)

# +
panel_four = sg.fromfile(
    f"{str(pca_association_path)}/category_box_plot/category_box_plot_pc1.svg"
)


panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.1
scale_y = 1.1

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(20, 1000)

# +
panel_five = sg.fromfile(
    pca_association_path / "category_box_plot/category_box_plot_pc2.svg"
)


panel_five_size = (
    np.round(float(panel_five.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_five.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.1
scale_y = 1.1

print(f"original: {panel_five_size}")
print(f"scaled:{(panel_five_size[0]*scale_x, panel_five_size[1]*scale_y)}")

panel_five = panel_five.getroot()
panel_five.scale(x=scale_x, y=scale_y)
panel_five.moveto(650, 1000)
# -

panel_one_label = sg.TextElement(20, 20, "A", size=22, weight="bold")
panel_one_image_label = sg.TextElement(250, 930, "PC 1", size=22, weight="bold")
panel_two_label = sg.TextElement(10, 520, "B", size=22, weight="bold")
panel_two_image_label = sg.TextElement(860, 930, "PC 2", size=22, weight="bold")
panel_three_label = sg.TextElement(640, 520, "C", size=22, weight="bold")
panel_four_label = sg.TextElement(10, 1000, "D", size=22, weight="bold")
panel_five_label = sg.TextElement(640, 1000, "E", size=22, weight="bold")

# +
figure_two = sg.SVGFigure(
    Unit(
        max(
            [
                panel_one_size[0],
                panel_two_size[0] + panel_three_size[0],
                panel_four_size[0] + panel_five_size[0],
            ]
        )
        - 100
    ),
    Unit(
        panel_one_size[1]
        + max(panel_two_size[1], panel_three_size[1])
        + max(panel_four_size[1], panel_five_size[1])
        + 150,
    ),
)

figure_two.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_five,
        panel_one_label,
        panel_one_image_label,
        panel_two_label,
        panel_two_image_label,
        panel_three_label,
        panel_four_label,
        panel_five_label,
    ]
)
display(SVG(figure_two.to_str()))
# -

# save generated SVG files
figure_two.save("output/figure_two_panels.svg")

# ## Figure 3

article_distance_path = Path("../biorxiv/article_distances/output/figures")

# +
panel_one = sg.fromfile(article_distance_path / "biorxiv_article_distance.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(20, 20)

# +
panel_two = sg.fromfile(article_distance_path / "distance_bin_accuracy.svg")

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(580, 20)

# +
panel_three = sg.fromfile(article_distance_path / "publication_rate.svg")

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.3
scale_y = 1.3

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(40, 420)
# -

panel_one_label = sg.TextElement(10, 20, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(560, 20, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=22, weight="bold")

figure_three = sg.SVGFigure(
    Unit(max([panel_one_size[0] + panel_two_size[0], panel_three_size[0]]) - 100),
    Unit(max(panel_one_size[1], panel_two_size[1]) + panel_three_size[1]),
)
figure_three.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_one_label,
        panel_two_label,
        panel_three_label,
    ]
)
display(SVG(figure_three.to_str()))

# save generated SVG files
figure_three.save("output/figure_three_panels.svg")

# ## Figure 4

publication_delay_path = Path("../biorxiv/publication_delay_experiment/output")
time_to_publication_path = Path("../biorxiv/time_to_publication/output")

# +
panel_one = sg.fromfile(time_to_publication_path / "preprint_category_halflife.svg")

panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x, panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(30, 20)

# +
panel_two = sg.fromfile(
    publication_delay_path / "version_count_vs_publication_time_violin.svg"
)

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled: {(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(730, 0)

# +
panel_three = sg.fromfile(
    publication_delay_path / "article_distance_vs_publication_time_hex.svg"
)

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1.2
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled: {(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(40, 390)
# -

panel_one_label = sg.TextElement(10, 30, "A", size=22, weight="bold")
panel_two_label = sg.TextElement(730, 30, "B", size=22, weight="bold")
panel_three_label = sg.TextElement(10, 420, "C", size=22, weight="bold")

figure_four = sg.SVGFigure(
    Unit(max([panel_one_size[0] + panel_two_size[0], panel_three_size[0]]) - 130),
    Unit(max(panel_one_size[1], panel_two_size[1]) + panel_three_size[1] - 200),
)
figure_four.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_one_label,
        panel_two_label,
        panel_three_label,
    ]
)
display(SVG(figure_four.to_str()))

# save generated SVG files
figure_four.save("output/figure_four_panels.svg")
